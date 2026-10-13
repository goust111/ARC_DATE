from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_coords, components_mask, paste


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

_BASE_SHAPES = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
]

_GEN_ATTEMPTS = 400
_GRID_H_RANGE = (18, 23)
_MIN_CANDIDATES = 4
_MAX_CANDIDATES = 6
_MAX_CANDIDATE_DIM = 5
_MAX_CANDIDATE_AREA = 10
_COL_GAP = 1


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    rows = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            rows.append(r)
    mask = mask[rows, :]

    cols = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            cols.append(c)
    return mask[:, cols]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


def _compressed_signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_key(_compress_runs(variant)) for variant in _variants(mask)}


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _horizontal_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[3] < b[1] or b[3] < a[1])


def _slide_by_bbox(
    objects: list[dict],
    height: int,
) -> list[dict]:
    placed: list[dict] = []
    top_group = sorted((obj for obj in objects if obj["out_color"] == RED), key=lambda obj: obj["input_r0"])
    for obj in top_group:
        dest_r0 = 1
        bbox = obj["input_bbox"]
        for other in placed:
            if other["bbox"][2] < bbox[0] and _horizontal_overlap(bbox, other["bbox"]):
                dest_r0 = max(dest_r0, other["bbox"][2] + 1)
        placed.append(
            {
                "mask": obj["mask"],
                "color": RED,
                "bbox": (dest_r0, obj["c0"], dest_r0 + obj["mask"].shape[0] - 1, obj["c0"] + obj["mask"].shape[1] - 1),
            }
        )

    bottom_group = sorted((obj for obj in objects if obj["out_color"] == BLUE), key=lambda obj: obj["input_r0"], reverse=True)
    for obj in bottom_group:
        h = obj["mask"].shape[0]
        dest_r0 = height - 1 - h
        bbox = obj["input_bbox"]
        for other in placed:
            if other["bbox"][0] > bbox[2] and _horizontal_overlap(bbox, other["bbox"]):
                dest_r0 = min(dest_r0, other["bbox"][0] - h)
        placed.append(
            {
                "mask": obj["mask"],
                "color": BLUE,
                "bbox": (dest_r0, obj["c0"], dest_r0 + h - 1, obj["c0"] + obj["mask"].shape[1] - 1),
            }
        )
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each yellow candidate by run-compressed shape against the grey prototype, then recolor and slide it to the matching border bar.

    Input structure:
    - Black background with a full-width red bar on row 0 and a full-width blue bar on the last row.
    - One grey 4-connected prototype object sits away from the border bars.
    - Four to six yellow 4-connected candidate objects are scattered in the interior.

    Rule:
    - Crop the grey prototype and every yellow candidate to their bounding boxes.
    - Run-compress rows and columns of each cropped mask, then compare each candidate against the prototype under all 8 rotation/reflection variants.
    - Recolor matching candidates red and non-matching candidates blue, erase the grey prototype, then slide red objects upward and blue objects downward.
    - Upward movers are processed by increasing original row; downward movers are processed by decreasing original row, and each object stops at the bar-adjacent row or just before an already stopped overlapping bbox.

    Key insight:
    - The grey object is only a hidden classifier: compression removes row/column stretching, and the resulting class decides both the recolor and the slide direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5
    assert np.all(g[0, :] == RED)
    assert np.all(g[h - 1, :] == BLUE)

    proto_comps = components_mask(g == GREY, connectivity=4)
    assert len(proto_comps) == 1
    proto_mask = _mask_from_coords(proto_comps[0])
    proto_sigs = _compressed_signature_set(proto_mask)

    candidate_comps = components_mask(g == YELLOW, connectivity=4)
    assert candidate_comps

    objects: list[dict] = []
    for coords in candidate_comps:
        mask = _mask_from_coords(coords)
        r0, c0, r1, c1 = bbox_coords(coords)
        is_match = _mask_key(_compress_runs(mask)) in proto_sigs
        objects.append(
            {
                "mask": mask,
                "c0": int(c0),
                "input_r0": int(r0),
                "input_bbox": (int(r0), int(c0), int(r1), int(c1)),
                "out_color": RED if is_match else BLUE,
            }
        )

    out = np.zeros_like(g)
    out[0, :] = RED
    out[h - 1, :] = BLUE

    for placed in _slide_by_bbox(objects, h):
        r0, c0, _r1, _c1 = placed["bbox"]
        paste(out, _paint_patch(placed["mask"], placed["color"]), r0, c0, transparent=BG)

    return out


def _choose_transform(mask: np.ndarray, require_nontrivial: bool) -> tuple[np.ndarray, bool]:
    options = []
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for flip in (False, True):
            variant = np.fliplr(rotated) if flip else rotated
            nontrivial = turns != 0 or flip
            if require_nontrivial and not nontrivial:
                continue
            options.append((variant.copy(), nontrivial))
    return random.choice(options)


def _stretch_options(mask: np.ndarray, require_extra: bool) -> list[tuple[np.ndarray, bool]]:
    options: list[tuple[np.ndarray, bool]] = []
    seen = set()
    row_choices = list(itertools.product((1, 2), repeat=mask.shape[0]))
    col_choices = list(itertools.product((1, 2), repeat=mask.shape[1]))
    for row_factors in row_choices:
        for col_factors in col_choices:
            used_extra = any(v == 2 for v in row_factors + col_factors)
            if require_extra and not used_extra:
                continue
            stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
            if stretched.shape[0] > _MAX_CANDIDATE_DIM or stretched.shape[1] > _MAX_CANDIDATE_DIM:
                continue
            if int(stretched.sum()) > _MAX_CANDIDATE_AREA:
                continue
            key = _mask_key(stretched)
            if key in seen:
                continue
            seen.add(key)
            options.append((stretched, used_extra))
    return options


def _choose_stretch(mask: np.ndarray, require_extra: bool) -> tuple[np.ndarray, bool]:
    options = _stretch_options(mask, require_extra=require_extra)
    if not options and require_extra:
        options = _stretch_options(mask, require_extra=False)
    if not options:
        raise ValueError("no stretch options")
    return random.choice(options)


def _signature_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return _mask_key(_compress_runs(a)) in _compressed_signature_set(b)


def _holeless(mask: np.ndarray) -> bool:
    return np.array_equal(binary_fill_holes(mask), mask)


def _choose_prototype() -> np.ndarray:
    candidates = [shape for shape in _BASE_SHAPES if 4 <= int(shape.sum()) <= 8 and _holeless(shape)]
    return random.choice(candidates).copy()


def _choose_distractor(proto: np.ndarray, require_extra: bool) -> tuple[np.ndarray, bool]:
    proto_sigs = _compressed_signature_set(proto)
    pool = _BASE_SHAPES[:]
    random.shuffle(pool)
    for base in pool:
        if _compressed_signature_set(base) & proto_sigs:
            continue
        transformed, _ = _choose_transform(base, require_nontrivial=False)
        stretched, used_extra = _choose_stretch(transformed, require_extra=require_extra)
        if _mask_key(_compress_runs(stretched)) not in proto_sigs:
            return stretched, used_extra
    raise ValueError("failed to find distractor")


def _assign_positions(widths: list[int], start: int, end: int) -> list[int] | None:
    if not widths:
        return []
    total = sum(widths) + _COL_GAP * (len(widths) - 1)
    available = end - start + 1
    if total > available:
        return None
    slack = available - total
    extras = np.random.multinomial(slack, np.ones(len(widths) + 1) / (len(widths) + 1))
    cols = []
    c = start + int(extras[0])
    for idx, width in enumerate(widths):
        cols.append(int(c))
        c += width
        if idx < len(widths) - 1:
            c += _COL_GAP + int(extras[idx + 1])
    return cols


def _expected_output_from_scene(height: int, width: int, objects: list[dict]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[0, :] = RED
    out[height - 1, :] = BLUE
    for placed in _slide_by_bbox(objects, height):
        r0, c0, _r1, _c1 = placed["bbox"]
        paste(out, _paint_patch(placed["mask"], placed["color"]), r0, c0, transparent=BG)
    return out


def _semantic_checks(grid: np.ndarray, out: np.ndarray, proto_mask: np.ndarray, objects: list[dict]) -> None:
    proto_sigs = _compressed_signature_set(proto_mask)
    raw_proto_variants = {_mask_key(variant) for variant in _variants(proto_mask)}

    matches = [obj for obj in objects if obj["match"]]
    mismatches = [obj for obj in objects if not obj["match"]]
    assert matches and mismatches
    assert any(obj["used_extra"] for obj in matches), "need at least one stretched match"
    assert any(obj["transform_nontrivial"] for obj in matches), "need at least one rotated/flipped match"
    assert any(obj["used_extra"] for obj in mismatches), "need at least one stretched distractor"

    match_rows = [obj["input_r0"] for obj in matches]
    mismatch_rows = [obj["input_r0"] for obj in mismatches]
    assert not (max(match_rows) < min(mismatch_rows) or max(mismatch_rows) < min(match_rows)), "row order should not reveal class"

    seen_cols = set()
    for obj in objects:
        bbox = obj["input_bbox"]
        for c in range(bbox[1], bbox[3] + 1):
            assert c not in seen_cols, "candidate bbox columns must be disjoint"
            seen_cols.add(c)
        sig_match = _mask_key(_compress_runs(obj["mask"])) in proto_sigs
        assert sig_match == obj["match"]

    assert any(_mask_key(obj["mask"]) not in raw_proto_variants for obj in matches), "at least one match must require compression"
    assert np.any(grid != out), "input and output should differ"
    assert np.any(out == RED) and np.any(out == BLUE)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        proto_mask = _choose_prototype()
        n_candidates = random.randint(_MIN_CANDIDATES, _MAX_CANDIDATES)
        n_matches = random.randint(1, n_candidates - 1)
        n_mismatches = n_candidates - n_matches

        candidates: list[dict] = []
        for idx in range(n_matches):
            transformed, nontrivial = _choose_transform(proto_mask, require_nontrivial=(idx == 1))
            stretched, used_extra = _choose_stretch(transformed, require_extra=(idx == 0))
            candidates.append(
                {
                    "mask": stretched,
                    "match": True,
                    "used_extra": used_extra,
                    "transform_nontrivial": nontrivial,
                }
            )

        for idx in range(n_mismatches):
            stretched, used_extra = _choose_distractor(proto_mask, require_extra=(idx == 0))
            candidates.append(
                {
                    "mask": stretched,
                    "match": False,
                    "used_extra": used_extra,
                    "transform_nontrivial": False,
                }
            )

        random.shuffle(candidates)
        left_count = n_candidates // 2
        left_objs = candidates[:left_count]
        right_objs = candidates[left_count:]
        left_widths = [obj["mask"].shape[1] for obj in left_objs]
        right_widths = [obj["mask"].shape[1] for obj in right_objs]

        proto_h, proto_w = proto_mask.shape
        left_need = sum(left_widths) + _COL_GAP * max(0, len(left_widths) - 1)
        right_need = sum(right_widths) + _COL_GAP * max(0, len(right_widths) - 1)
        min_w = max(18, left_need + proto_w + right_need + 4)
        if min_w > 30:
            continue
        width = random.randint(min_w, min(30, min_w + 4))
        proto_c0_min = left_need + 2
        proto_c0_max = width - proto_w - right_need - 2
        if proto_c0_min > proto_c0_max:
            continue
        center = width // 2 - proto_w // 2
        proto_c0 = min(range(proto_c0_min, proto_c0_max + 1), key=lambda c: (abs(c - center), random.random()))

        left_cols = _assign_positions(left_widths, 1, proto_c0 - 2)
        right_cols = _assign_positions(right_widths, proto_c0 + proto_w + 2, width - 2)
        if left_cols is None or right_cols is None:
            continue

        height = random.randint(max(_GRID_H_RANGE[0], proto_h + 8), _GRID_H_RANGE[1] - 1)
        max_cand_h = max(obj["mask"].shape[0] for obj in candidates)
        height = max(height, max_cand_h + 6)
        if height > 30:
            continue

        proto_r0_max = min(7, height - proto_h - 4)
        if proto_r0_max < 3:
            continue
        proto_r0 = random.randint(3, proto_r0_max)

        for obj, c0 in zip(left_objs, left_cols):
            obj["c0"] = int(c0)
        for obj, c0 in zip(right_objs, right_cols):
            obj["c0"] = int(c0)

        row_ok = True
        for obj in candidates:
            obj_h = obj["mask"].shape[0]
            if obj["match"]:
                low = 2
                high = height - obj_h - 3
            else:
                low = 1
                high = height - obj_h - 2
            if low > high:
                row_ok = False
                break
            obj["input_r0"] = random.randint(low, high)
            r0 = obj["input_r0"]
            c0 = obj["c0"]
            obj["input_bbox"] = (r0, c0, r0 + obj_h - 1, c0 + obj["mask"].shape[1] - 1)
            obj["out_color"] = RED if obj["match"] else BLUE
        if not row_ok:
            continue

        grid = np.zeros((height, width), dtype=int)
        grid[0, :] = RED
        grid[height - 1, :] = BLUE
        paste(grid, _paint_patch(proto_mask, GREY), proto_r0, proto_c0, transparent=BG)

        for obj in candidates:
            paste(grid, _paint_patch(obj["mask"], YELLOW), obj["input_r0"], obj["c0"], transparent=BG)

        expected = _expected_output_from_scene(height, width, candidates)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        try:
            _semantic_checks(grid, actual, proto_mask, candidates)
        except AssertionError:
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
