from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SIZE_CHOICES = (2, 3, 4)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _build_object_from_square(
    shape: tuple[int, int], r0: int, c0: int, size: int, object_id: int
) -> dict:
    h, w = shape
    margin = size // 2
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    fr0 = r0 - margin - 1
    fc0 = c0 - margin - 1
    frame_side = 2 * size + 2
    fr1 = fr0 + frame_side - 1
    fc1 = fc0 + frame_side - 1

    square_mask = np.zeros(shape, dtype=bool)
    square_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    frame_mask = np.zeros(shape, dtype=bool)
    if 0 <= fr0 < h:
        c_lo = max(fc0, 0)
        c_hi = min(fc1, w - 1)
        if c_lo <= c_hi:
            frame_mask[fr0, c_lo : c_hi + 1] = True
    if 0 <= fr1 < h:
        c_lo = max(fc0, 0)
        c_hi = min(fc1, w - 1)
        if c_lo <= c_hi:
            frame_mask[fr1, c_lo : c_hi + 1] = True
    if 0 <= fc0 < w:
        r_lo = max(fr0, 0)
        r_hi = min(fr1, h - 1)
        if r_lo <= r_hi:
            frame_mask[r_lo : r_hi + 1, fc0] = True
    if 0 <= fc1 < w:
        r_lo = max(fr0, 0)
        r_hi = min(fr1, h - 1)
        if r_lo <= r_hi:
            frame_mask[r_lo : r_hi + 1, fc1] = True

    return {
        "id": int(object_id),
        "size": int(size),
        "area": int(size * size),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "square_mask": square_mask,
        "frame_mask": frame_mask,
    }


def _extract_square_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = sorted(set(map(int, g.ravel())) - {bg})
    assert fg_colors, "expected at least one foreground object"

    objects: list[dict] = []
    next_id = 0
    for color in fg_colors:
        for coords in components_mask(g == color):
            r0, c0, r1, c1 = bbox_coords(coords)
            size = r1 - r0 + 1
            assert size == c1 - c0 + 1, "foreground components must be square"
            assert coords.shape[0] == size * size, "foreground squares must be solid"
            objects.append(_build_object_from_square(g.shape, r0, c0, size, next_id))
            next_id += 1
    assert objects, "no square objects found"
    return objects


def _assign_ranks(objects: list[dict]) -> list[dict]:
    ordered = sorted(objects, key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1]))
    for rank, obj in enumerate(ordered, start=1):
        obj["rank"] = int(rank)
    return ordered


def _compute_ray_cells(
    shape: tuple[int, int], objects: list[dict], use_other_frames: bool
) -> dict[int, list[tuple[int, int]]]:
    h, w = shape
    rays: dict[int, list[tuple[int, int]]] = {}
    for obj in objects:
        obstacle = np.zeros(shape, dtype=bool)
        for other in objects:
            if other["id"] == obj["id"]:
                continue
            obstacle |= other["square_mask"]
            if use_other_frames:
                obstacle |= other["frame_mask"]

        r0, c0, r1, c1 = obj["bbox"]
        cells: list[tuple[int, int]] = []
        for c in range(c0, c1 + 1):
            for r in range(r1 + 1, h):
                if c < 0 or c >= w:
                    break
                if obstacle[r, c]:
                    break
                cells.append((r, c))
        rays[obj["id"]] = cells
    return rays


def _render_output(
    shape: tuple[int, int],
    objects: list[dict],
    use_other_frames: bool = True,
    use_rank_colors: bool = True,
) -> np.ndarray:
    objects = _assign_ranks(objects)
    out = np.full(shape, BLACK, dtype=int)

    for obj in objects:
        out[obj["frame_mask"]] = GREEN
    for obj in objects:
        out[obj["square_mask"]] = YELLOW

    ray_cells = _compute_ray_cells(shape, objects, use_other_frames=use_other_frames)
    for obj in sorted(objects, key=lambda item: item["rank"], reverse=True):
        color = BLUE if (not use_rank_colors or obj["rank"] % 2 == 1) else RED
        for r, c in ray_cells[obj["id"]]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw doubled green frames around yellow squares, then project ranked downward rays blocked by other frames.

    Input structure:
    - The grid has a black background and several solid yellow square objects.
    - The squares have distinct areas and do not overlap.
    - No green, blue, or red construction marks appear in the input.

    Rule:
    - For each yellow square, draw a hollow green frame whose interior side length is twice the square side, with the square centered inside using floor(side/2) top-left padding.
    - Rank the squares by decreasing area; odd ranks use blue rays and even ranks use red rays.
    - From each square's bottom edge, extend a same-width downward ray one column at a time; each column stops before the first non-black cell belonging to another square or another frame, while the square's own frame does not block its own ray.
    - If multiple rays want the same cell, the higher-ranked square keeps that cell.

    Key insight:
    - The output depends on two separate cues at once: object size fixes ray color via ranking, while the geometry of other objects' doubled frames determines how far each ray column can descend.
    """
    objects = _extract_square_objects(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid).shape, objects)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _build_object_from_frame(
    shape: tuple[int, int], fr0: int, fc0: int, size: int, object_id: int
) -> dict:
    margin = size // 2
    r0 = fr0 + 1 + margin
    c0 = fc0 + 1 + margin
    return _build_object_from_square(shape, r0, c0, size, object_id)


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for obj in objects:
        grid[obj["square_mask"]] = YELLOW
    return grid


def _column_lengths(cells: list[tuple[int, int]]) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for _r, c in cells:
        lengths[c] = lengths.get(c, 0) + 1
    return lengths


def _sample_layout() -> list[dict] | None:
    h = random.randint(19, 27)
    w = random.randint(24, 30)
    shape = (h, w)
    source_size, blocker_size, free_size = random.sample(SIZE_CHOICES, 3)
    source_frame = 2 * source_size + 2
    blocker_frame = 2 * blocker_size + 2
    free_frame = 2 * free_size + 2
    overlap = random.randint(1, source_size - 1)
    side = random.choice(("left", "right"))
    vertical_gap = random.randint(1, 3)
    horizontal_gap = random.randint(2, 4)
    upper_limit = (2 * h) // 3

    blocker_row_min = source_frame + vertical_gap
    blocker_row_max = min(h - blocker_frame, upper_limit - (blocker_size // 2) - blocker_size)
    if blocker_row_min > blocker_row_max:
        return None
    blocker_fr0 = random.randint(blocker_row_min, blocker_row_max)

    source_row_max = min(
        blocker_fr0 - source_frame - vertical_gap,
        upper_limit - (source_size // 2) - source_size,
    )
    if source_row_max < 0:
        return None
    source_fr0 = random.randint(0, source_row_max)

    base_candidates: list[tuple[int, int, list[int]]] = []
    for blocker_fc0 in range(0, w - blocker_frame + 1):
        if side == "left":
            source_c0 = blocker_fc0 - (source_size - overlap)
            source_fc0 = source_c0 - (source_size // 2) - 1
            free_fc_choices = list(range(blocker_fc0 + blocker_frame + horizontal_gap, w - free_frame + 1))
        else:
            source_c0 = blocker_fc0 + blocker_frame - overlap
            source_fc0 = source_c0 - (source_size // 2) - 1
            free_fc_choices = list(range(0, blocker_fc0 - horizontal_gap - free_frame + 1))
        if source_fc0 < 0 or source_fc0 + source_frame > w:
            continue
        if not free_fc_choices:
            continue
        base_candidates.append((blocker_fc0, source_fc0, free_fc_choices))
    if not base_candidates:
        return None

    blocker_fc0, source_fc0, free_fc_choices = random.choice(base_candidates)
    free_fc0 = random.choice(free_fc_choices)

    source_box = (source_fr0, source_fc0, source_fr0 + source_frame - 1, source_fc0 + source_frame - 1)
    blocker_box = (
        blocker_fr0,
        blocker_fc0,
        blocker_fr0 + blocker_frame - 1,
        blocker_fc0 + blocker_frame - 1,
    )

    free_row_max = min(h - free_frame, upper_limit - (free_size // 2) - free_size)
    if free_row_max < 0:
        return None
    free_row_choices = []
    for free_fr0 in range(0, free_row_max + 1):
        free_box = (free_fr0, free_fc0, free_fr0 + free_frame - 1, free_fc0 + free_frame - 1)
        if _boxes_separated(free_box, source_box, pad=1) and _boxes_separated(free_box, blocker_box, pad=1):
            free_row_choices.append(free_fr0)
    if not free_row_choices:
        return None
    free_fr0 = random.choice(free_row_choices)

    objects = [
        _build_object_from_frame(shape, source_fr0, source_fc0, source_size, 0),
        _build_object_from_frame(shape, blocker_fr0, blocker_fc0, blocker_size, 1),
        _build_object_from_frame(shape, free_fr0, free_fc0, free_size, 2),
    ]

    frame_boxes = [obj["frame_bbox"] for obj in objects]
    for i in range(len(frame_boxes)):
        for j in range(i + 1, len(frame_boxes)):
            if not _boxes_separated(frame_boxes[i], frame_boxes[j], pad=1):
                return None
    return objects


def _validate_candidate(objects: list[dict]) -> bool:
    shape = objects[0]["square_mask"].shape
    input_grid = _render_input(shape, objects)
    oracle = _render_output(shape, objects)
    if not np.array_equal(main(input_grid), oracle):
        return False

    oracle_no_frame = _render_output(shape, objects, use_other_frames=False)
    oracle_no_rank = _render_output(shape, objects, use_rank_colors=False)
    if np.array_equal(oracle, oracle_no_frame):
        return False
    if np.array_equal(oracle, oracle_no_rank):
        return False

    ray_cells = _compute_ray_cells(shape, _assign_ranks(objects), use_other_frames=True)
    ray_cells_no_frame = _compute_ray_cells(shape, _assign_ranks(objects), use_other_frames=False)

    source_obj = next(obj for obj in objects if obj["id"] == 0)
    blocker_obj = next(obj for obj in objects if obj["id"] == 1)
    source_lengths = _column_lengths(ray_cells[source_obj["id"]])
    source_lengths_no_frame = _column_lengths(ray_cells_no_frame[source_obj["id"]])
    source_cols = range(source_obj["bbox"][1], source_obj["bbox"][3] + 1)
    if not any(source_lengths.get(c, 0) < source_lengths_no_frame.get(c, 0) for c in source_cols):
        return False
    if not any(source_lengths.get(c, 0) == shape[0] - 1 - source_obj["bbox"][2] for c in source_cols):
        return False

    blocker_span = range(blocker_obj["frame_bbox"][1], blocker_obj["frame_bbox"][3] + 1)
    if len(set(source_cols) & set(blocker_span)) == 0:
        return False

    if RED not in oracle or BLUE not in oracle:
        return False
    if sum(len(cells) for cells in ray_cells.values()) < 8:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        objects = _sample_layout()
        if objects is None:
            continue
        if _validate_candidate(objects):
            return _render_input(objects[0]["square_mask"].shape, objects)
    raise RuntimeError("failed to generate a valid input")
