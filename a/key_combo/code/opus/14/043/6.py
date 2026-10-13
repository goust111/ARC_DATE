from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SCORE_TO_COLOR = {2: BLUE, 1: YELLOW, 0: RED}
GEN_RETRIES = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


DOUBLE_TEMPLATES = [
    _mask(["##", "##"]),
    _mask(["###", "###"]),
    _mask([".#.", "###", ".#."]),
    _mask(["###", "#.#", "###"]),
    _mask([".##.", "####", "####", ".##."]),
]

SINGLE_TEMPLATES = [
    _mask(["###", ".#."]),
    _mask(["###", ".#.", ".#."]),
    _mask(["#.#", "###"]),
    _mask(["####", ".##."]),
    _mask([".##.", "####", ".##."]),
]

NONE_TEMPLATES = [
    _mask(["#.", "#.", "##"]),
    _mask(["##.", ".##"]),
    _mask(["##.", ".#.", ".##"]),
    _mask(["#..", "###", "..#"]),
    _mask(["###.", ".##.", "..##"]),
]

SHAPE_CATALOG = {2: DOUBLE_TEMPLATES, 1: SINGLE_TEMPLATES, 0: NONE_TEMPLATES}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return _trim_mask(out)


def _symmetry_details(mask: np.ndarray) -> tuple[bool, bool, int]:
    lr = np.array_equal(mask, np.fliplr(mask))
    ud = np.array_equal(mask, np.flipud(mask))
    score = 2 if lr and ud else 1 if lr or ud else 0
    return lr, ud, score


def _extract_green_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    objects: list[dict] = []

    for r in range(H):
        for c in range(W):
            if g[r, c] != GREEN or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == GREEN:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            lr, ud, score = _symmetry_details(mask)
            objects.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "lr_sym": lr,
                    "ud_sym": ud,
                    "score": score,
                    "area": len(coords),
                }
            )

    return objects


def _render_objects(objects: list[dict]) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: (-obj["score"], obj["bbox"][0], obj["bbox"][1]))
    height = max(obj["mask"].shape[0] for obj in ordered)
    width = sum(obj["mask"].shape[1] for obj in ordered) + len(ordered) - 1
    out = np.zeros((height, width), dtype=int)

    c0 = 0
    for obj in ordered:
        mask = obj["mask"]
        h, w = mask.shape
        out[:h, c0 : c0 + w][mask] = SCORE_TO_COLOR[obj["score"]]
        c0 += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort cropped green sprites by their mirror-symmetry grade and concatenate them into one row.

    Input structure:
    - The grid has black background cells and several disconnected green 4-connected sprites.
    - Each sprite fits inside a tight bounding box no larger than 4x4.
    - Different sprites may be doubly symmetric, singly symmetric under exactly one flip, or asymmetric.

    Rule:
    - Extract every green sprite as a tight bbox mask.
    - Test each cropped mask against left-right and up-down flips.
    - Assign symmetry scores: both flips match -> 2, exactly one flip matches -> 1, neither matches -> 0.
    - Recolor score-2 sprites blue, score-1 sprites yellow, and score-0 sprites red.
    - Sort sprites by score descending, breaking ties by the original bbox top-left position in row-major order.
    - Output the recolored cropped sprites as a single top-aligned horizontal strip with one black column between neighbors.

    Key insight:
    - Mirroring is only a diagnostic test here: the flip results are never drawn, but the detected symmetry simultaneously determines both recoloring and concatenation order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    objects = _extract_green_objects(g)
    assert objects, "expected at least one green sprite"
    return _render_objects(objects)


def _sample_sprite(score: int, required_axis: str | None = None) -> dict:
    for _ in range(200):
        base = random.choice(SHAPE_CATALOG[score])
        mask = _transform_mask(base)
        lr, ud, actual_score = _symmetry_details(mask)
        if actual_score != score:
            continue
        if required_axis == "lr" and not (lr and not ud):
            continue
        if required_axis == "ud" and not (ud and not lr):
            continue
        return {
            "mask": mask,
            "score": score,
            "lr_sym": lr,
            "ud_sym": ud,
            "area": int(mask.sum()),
        }
    raise RuntimeError(f"failed to sample sprite for score={score}")


def _choose_scores() -> list[int]:
    count = random.randint(4, 6)
    scores = [2, 1, 0]
    while len(scores) < count:
        scores.append(random.choice([2, 1, 0]))
    random.shuffle(scores)
    return scores


def _build_object_specs() -> list[dict]:
    scores = _choose_scores()
    single_count = scores.count(1)
    axis_needs = ["lr", "ud"] if single_count >= 2 else []
    random.shuffle(axis_needs)

    objects = []
    for score in scores:
        required_axis = axis_needs.pop() if score == 1 and axis_needs else None
        obj = _sample_sprite(score, required_axis=required_axis)
        objects.append(obj)
    return objects


def _canvas_shape(n_objects: int) -> tuple[int, int]:
    min_side = {4: 18, 5: 20, 6: 22}[n_objects]
    return random.randint(min_side, 24), random.randint(min_side, 24)


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    H, W = occupied.shape
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(mask):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    H, W = _canvas_shape(len(objects))
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    placement_order = sorted(
        range(len(objects)),
        key=lambda i: (objects[i]["mask"].shape[0] * objects[i]["mask"].shape[1], objects[i]["area"]),
        reverse=True,
    )

    for idx in placement_order:
        obj = objects[idx]
        mask = obj["mask"]
        h, w = mask.shape
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _can_place(occupied, mask, r0, c0):
                continue
            grid[r0 : r0 + h, c0 : c0 + w][mask] = GREEN
            for dr, dc in np.argwhere(mask):
                occupied[r0 + int(dr), c0 + int(dc)] = True
            obj["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
            placed = True
            break
        if not placed:
            return None

    return grid


def _semantic_checks(objects: list[dict]) -> bool:
    score_counts = Counter(obj["score"] for obj in objects)
    if set(score_counts) != {0, 1, 2}:
        return False
    if max(score_counts.values()) < 2:
        return False

    row_order = sorted(range(len(objects)), key=lambda i: (objects[i]["bbox"][0], objects[i]["bbox"][1]))
    target_order = sorted(range(len(objects)), key=lambda i: (-objects[i]["score"], objects[i]["bbox"][0], objects[i]["bbox"][1]))
    if row_order == target_order:
        return False

    area_order = sorted(range(len(objects)), key=lambda i: (-objects[i]["area"], objects[i]["bbox"][0], objects[i]["bbox"][1]))
    if area_order == target_order:
        return False

    shared_area = any(
        objects[i]["area"] == objects[j]["area"] and objects[i]["score"] != objects[j]["score"]
        for i in range(len(objects))
        for j in range(i + 1, len(objects))
    )
    shared_bbox = any(
        objects[i]["mask"].shape == objects[j]["mask"].shape and objects[i]["score"] != objects[j]["score"]
        for i in range(len(objects))
        for j in range(i + 1, len(objects))
    )
    if not (shared_area or shared_bbox):
        return False

    singles = [obj for obj in objects if obj["score"] == 1]
    if len(singles) >= 2:
        has_lr_only = any(obj["lr_sym"] and not obj["ud_sym"] for obj in singles)
        has_ud_only = any(obj["ud_sym"] and not obj["lr_sym"] for obj in singles)
        if not (has_lr_only and has_ud_only):
            return False

    return True


def _expected_from_objects(objects: list[dict]) -> np.ndarray:
    logical = []
    for obj in objects:
        logical.append(
            {
                "mask": obj["mask"].copy(),
                "score": obj["score"],
                "bbox": obj["bbox"],
            }
        )
    return _render_objects(logical)


def _validate_candidate(grid: np.ndarray, objects: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not _semantic_checks(objects):
        return False

    extracted = _extract_green_objects(grid)
    if len(extracted) != len(objects):
        return False

    found_scores = Counter(obj["score"] for obj in extracted)
    expected_scores = Counter(obj["score"] for obj in objects)
    if found_scores != expected_scores:
        return False

    found_bboxes = sorted(obj["bbox"] for obj in extracted)
    expected_bboxes = sorted(obj["bbox"] for obj in objects)
    if found_bboxes != expected_bboxes:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    recolor_set = set(int(v) for v in expected.flat if v != BLACK)
    if recolor_set != {BLUE, YELLOW, RED}:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        objects = _build_object_specs()
        grid = _place_objects(objects)
        if grid is None:
            continue
        expected = _expected_from_objects(objects)
        if _validate_candidate(grid, objects, expected):
            return grid
    raise RuntimeError("failed to generate a valid symmetry-sorting ARC input")
