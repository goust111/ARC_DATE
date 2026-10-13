from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, get_bg, paste, random_free_location


OUTPUT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
BG_CANDIDATES = (Color.BLACK, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
INPUT_OBJECT_CANDIDATES = (Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
BASE_BOX_MIN = 3
BASE_BOX_MAX = 5
BASE_AREA_MIN = 5
BASE_AREA_MAX = 8
REAL_REF_COUNT = (3, 5)
REAL_OTHER_COUNT = (1, 2)
MIRROR_COUNT = (1, 3)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
SHAPE_RETRY_LIMIT = 200
CASE_RETRY_LIMIT = 400


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), tuple(map(int, arr.ravel()))


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k).copy() for k in range(4)]


def _canonical_rotation(mask: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, ...]]]:
    rots = _rotations(mask)
    sigs = [_mask_signature(rot) for rot in rots]
    best = min(range(4), key=lambda k: sigs[k])
    return rots[best], sigs[best]


def _rotation_index(mask: np.ndarray, canonical: np.ndarray) -> int:
    for k, rot in enumerate(_rotations(canonical)):
        if np.array_equal(rot, mask):
            return k
    raise AssertionError("mask is not a rotation of the canonical sprite")


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=4)
    assert comps, "expected at least one foreground object"

    out = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        full_mask = np.zeros(grid.shape, dtype=bool)
        full_mask[coords[:, 0], coords[:, 1]] = True
        mask = crop_bbox(full_mask, (int(r0), int(c0), int(r1), int(c1)))
        out.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
            }
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the majority-handed rotation family by steps from its modal pose, and erase mirrored impostors.

    Input structure:
    - The background is the most common color, and every foreground object is the same monochrome sprite.
    - Genuine objects are 4-connected copies of one chiral base sprite under 0, 90, 180, or 270 degree rotation.
    - A few foreground objects are mirrored copies of that sprite, so they belong to the opposite handedness.

    Rule:
    - Split the objects into the two handedness classes induced by rotation-only canonical forms, and keep the class with strictly more members.
    - Within the kept class, find the unique most frequent orientation and use it as the reference pose.
    - Recolor kept objects without moving them: same pose -> blue, clockwise 90 -> red, clockwise 180 -> green, clockwise 270 -> yellow; erase every mirrored-class object to background.

    Key insight:
    - Orientation is hidden in shape pose alone: a rotation-only canonical signature separates chirality, and the modal pose inside the majority chirality becomes the data-bearing reference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    records = _component_records(g, bg)

    pair_signature = None
    handed_counts: Counter = Counter()
    by_handedness: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = {}

    for rec in records:
        mask = rec["mask"]
        rot_canon, rot_sig = _canonical_rotation(mask)
        mirror_canon, mirror_sig = _canonical_rotation(np.fliplr(mask))
        rec["rot_canon"] = rot_canon
        rec["rot_sig"] = rot_sig
        rec["mirror_sig"] = mirror_sig

        pair = tuple(sorted((rot_sig, mirror_sig)))
        if pair_signature is None:
            pair_signature = pair
        else:
            assert pair == pair_signature, "all objects must be the same sprite up to rotation or reflection"

        handed_counts[rot_sig] += 1
        by_handedness.setdefault(rot_sig, []).append(rec)

    ranked_hands = handed_counts.most_common()
    assert ranked_hands, "expected handedness counts"
    if len(ranked_hands) > 1:
        assert ranked_hands[0][1] > ranked_hands[1][1], "expected a unique majority handedness"
    keep_sig = ranked_hands[0][0]
    kept = by_handedness[keep_sig]

    canonical = kept[0]["rot_canon"]
    orientations = []
    for rec in kept:
        assert np.array_equal(rec["rot_canon"], canonical), "kept handedness must share one canonical rotation"
        rec["orientation"] = _rotation_index(rec["mask"], canonical)
        orientations.append(rec["orientation"])

    pose_counts = Counter(orientations).most_common()
    assert pose_counts, "expected kept orientations"
    if len(pose_counts) > 1:
        assert pose_counts[0][1] > pose_counts[1][1], "expected a unique modal orientation"
    ref_orientation = int(pose_counts[0][0])

    out = np.full_like(g, bg)
    for rec in kept:
        step = (ref_orientation - int(rec["orientation"])) % 4
        color = int(OUTPUT_COLORS[step])
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _is_valid_base(mask: np.ndarray) -> bool:
    mask = _trim_mask(mask)
    h, w = mask.shape
    if not (BASE_BOX_MIN <= h <= BASE_BOX_MAX and BASE_BOX_MIN <= w <= BASE_BOX_MAX):
        return False
    area = int(mask.sum())
    if not (BASE_AREA_MIN <= area <= BASE_AREA_MAX):
        return False
    if len(components_mask(mask, connectivity=4)) != 1:
        return False
    if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
        return False
    if any(np.array_equal(np.fliplr(mask), np.rot90(mask, k)) for k in range(4)):
        return False
    return True


def _random_base_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRY_LIMIT):
        h = random.randint(BASE_BOX_MIN, BASE_BOX_MAX)
        w = random.randint(BASE_BOX_MIN, BASE_BOX_MAX)
        area = random.randint(BASE_AREA_MIN, min(BASE_AREA_MAX, h * w - 1))

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        cells = [seed]
        while len(cells) < area:
            r, c = random.choice(cells)
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                continue
            pick = random.choice(neighbors)
            mask[pick] = True
            cells.append(pick)

        mask = _trim_mask(mask)
        if _is_valid_base(mask):
            return mask
    raise RuntimeError("failed to generate a chiral non-symmetric base sprite")


def _paint_patch(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(fg)
    return patch


def _sample_real_counts() -> tuple[int, list[int]]:
    ref = random.randrange(4)
    counts = [random.randint(*REAL_OTHER_COUNT) for _ in range(4)]
    counts[ref] = random.randint(*REAL_REF_COUNT)
    if any(counts[ref] <= counts[k] for k in range(4) if k != ref):
        return _sample_real_counts()
    return ref, counts


def _build_expected(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        if spec["kind"] == "mirror":
            continue
        paste(out, spec["output_patch"], spec["r0"], spec["c0"], transparent=bg)
    return out


def _check_case_semantics(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    bg = int(meta["bg"])
    obj = int(meta["obj"])
    specs = meta["objects"]

    if not np.all((inp == bg) | (inp == obj)):
        return False

    comps = components_mask(inp != bg, connectivity=4)
    if len(comps) != len(specs):
        return False

    real_specs = [spec for spec in specs if spec["kind"] == "real"]
    mirror_specs = [spec for spec in specs if spec["kind"] == "mirror"]
    if not real_specs or not mirror_specs:
        return False
    if len(real_specs) <= len(mirror_specs):
        return False

    orientation_counts = Counter(spec["orientation"] for spec in real_specs).most_common()
    if len(orientation_counts) < 2 or orientation_counts[0][1] <= orientation_counts[1][1]:
        return False
    if {spec["orientation"] for spec in real_specs} != {0, 1, 2, 3}:
        return False

    actual_boxes = sorted(
        (tuple(map(int, coords.min(axis=0))) + tuple(map(int, coords.max(axis=0))) for coords in comps)
    )
    spec_boxes = sorted(
        (int(spec["r0"]), int(spec["c0"]), int(spec["r0"] + spec["input_patch"].shape[0] - 1), int(spec["c0"] + spec["input_patch"].shape[1] - 1))
        for spec in specs
    )
    if actual_boxes != spec_boxes:
        return False

    for spec in specs:
        h, w = spec["input_patch"].shape
        in_patch = inp[spec["r0"] : spec["r0"] + h, spec["c0"] : spec["c0"] + w]
        if not np.array_equal(in_patch, spec["input_patch"]):
            return False
        out_patch = expected[spec["r0"] : spec["r0"] + h, spec["c0"] : spec["c0"] + w]
        if spec["kind"] == "mirror":
            if not np.all(out_patch == bg):
                return False
        elif not np.array_equal(out_patch, spec["output_patch"]):
            return False

    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    bg = int(random.choice(BG_CANDIDATES))
    obj_candidates = [c for c in INPUT_OBJECT_CANDIDATES if int(c) != bg and int(c) not in OUTPUT_COLORS]
    obj = int(random.choice(obj_candidates))

    base = _random_base_mask()
    mirror = np.fliplr(base)

    ref_orientation, real_counts = _sample_real_counts()
    mirror_count = random.randint(*MIRROR_COUNT)

    specs = []
    for orientation, count in enumerate(real_counts):
        patch_mask = np.rot90(base, orientation)
        input_patch = _paint_patch(patch_mask, obj, bg)
        out_color = int(OUTPUT_COLORS[(ref_orientation - orientation) % 4])
        output_patch = _paint_patch(patch_mask, out_color, bg)
        for _ in range(count):
            specs.append(
                {
                    "kind": "real",
                    "orientation": int(orientation),
                    "input_patch": input_patch,
                    "output_patch": output_patch,
                }
            )

    for _ in range(mirror_count):
        orientation = random.randrange(4)
        patch_mask = np.rot90(mirror, orientation)
        specs.append(
            {
                "kind": "mirror",
                "orientation": int(orientation),
                "input_patch": _paint_patch(patch_mask, obj, bg),
                "output_patch": np.full(patch_mask.shape, bg, dtype=int),
            }
        )

    random.shuffle(specs)

    max_h = max(spec["input_patch"].shape[0] for spec in specs)
    max_w = max(spec["input_patch"].shape[1] for spec in specs)
    span = int(np.ceil(np.sqrt(len(specs))) * (max(max_h, max_w) + 2))
    side_min = max(GRID_SIDE_MIN, span)
    if side_min > GRID_SIDE_MAX:
        raise RuntimeError("sprite packing bounds exceed ARC size limit")

    h = random.randint(side_min, GRID_SIDE_MAX)
    w = random.randint(side_min, GRID_SIDE_MAX)
    inp = np.full((h, w), bg, dtype=int)
    layout = np.full((h, w), bg, dtype=int)

    for spec in specs:
        patch = spec["input_patch"]
        footprint = np.full(patch.shape, obj, dtype=int)
        try:
            r0, c0 = random_free_location(
                layout,
                footprint,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError as exc:
            raise RuntimeError("placement failed") from exc
        paste(layout, footprint, r0, c0, transparent=bg)
        paste(inp, patch, r0, c0, transparent=bg)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)

    meta = {
        "bg": bg,
        "obj": obj,
        "base": base,
        "ref_orientation": int(ref_orientation),
        "real_counts": tuple(map(int, real_counts)),
        "objects": specs,
    }
    expected = _build_expected(inp.shape, bg, specs)
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(CASE_RETRY_LIMIT):
        try:
            inp, expected, meta = _generate_case()
        except RuntimeError:
            continue
        if not _check_case_semantics(inp, expected, meta):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid orientation-as-data input")
