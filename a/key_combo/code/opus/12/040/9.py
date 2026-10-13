from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
BLUE = 1
RED = 2
CRACK = 7
GEN_RETRIES = 500


def _swap_blue_red(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    blue_mask = patch == BLUE
    red_mask = patch == RED
    out[blue_mask] = RED
    out[red_mask] = BLUE
    return out


def _detect_crack_orientation(local: np.ndarray) -> str:
    crack = np.argwhere(local == CRACK)
    assert crack.size > 0, "each object must contain an orange crack line"

    rows = len(np.unique(crack[:, 0]))
    cols = len(np.unique(crack[:, 1]))
    height, width = local.shape

    if rows == height and cols < width:
        return "vertical"
    if cols == width and rows < height:
        return "horizontal"
    if width > height:
        return "vertical"
    if height > width:
        return "horizontal"
    raise AssertionError("crack orientation must be uniquely determined")


def _split_object(local: np.ndarray) -> tuple[str, np.ndarray, dict, dict]:
    orientation = _detect_crack_orientation(local)

    erased = local.copy()
    erased[erased == CRACK] = BG
    halves = find_components(
        erased,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=[BLUE, RED],
    )
    assert len(halves) == 2, "erasing the crack must split the object into two halves"

    for half in halves:
        values = erased[half["coords"][:, 0], half["coords"][:, 1]]
        assert np.any(values == BLUE) and np.any(values == RED), "each half must contain both colors"

    if orientation == "vertical":
        halves = sorted(halves, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    else:
        halves = sorted(halves, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    return orientation, erased, halves[0], halves[1]


def _transform_local_object(local: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]]]:
    orientation, erased, half_a, half_b = _split_object(local)

    a_patch = crop_bbox(erased, half_a["bbox"])
    b_patch = crop_bbox(erased, half_b["bbox"])
    mirrored_b = np.fliplr(b_patch) if orientation == "vertical" else np.flipud(b_patch)

    ar0, ac0, ar1, ac1 = map(int, half_a["bbox"])
    a_swapped = _swap_blue_red(a_patch)
    b_swapped = _swap_blue_red(mirrored_b)
    placements = [
        (a_swapped, ar0, ac0),
        (b_swapped, ar1 + 1, ac1 + 1),
    ]

    max_r = max(r0 + patch.shape[0] for patch, r0, _c0 in placements)
    max_c = max(c0 + patch.shape[1] for patch, _r0, c0 in placements)
    out = np.full((max_r, max_c), BG, dtype=int)
    for patch, r0, c0 in placements:
        paste(out, patch, r0, c0, transparent=BG)
    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each crack-marked two-color object, mirror one half, rejoin the halves diagonally, and swap blue/red in both halves.

    Input structure:
    - The grid has a black background and 1 or 2 non-touching multicolor foreground objects.
    - Each object uses blue and red pixels plus a 1-pixel orange crack line that fully spans the object's shorter bbox axis.
    - Removing the orange line splits the object into exactly two 4-connected halves: top/bottom for a horizontal crack or left/right for a vertical crack.

    Rule:
    - For each object, erase the orange crack line to obtain halves A and B, where A is the upper half for a horizontal crack or the left half for a vertical crack.
    - Mirror B across its own bbox center along the crack direction: horizontal crack -> flip B vertically; vertical crack -> flip B horizontally.
    - Keep A fixed and place the mirrored B so that B's bbox top-left is one cell down-right from A's bbox bottom-right.
    - Swap blue and red in both halves, and leave all other cells black.

    Key insight:
    - The crack line determines both how the object splits and which mirror axis to use, after which the mirrored second half is reassembled diagonally instead of returning to its original side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg = get_bg(g)
    assert int(bg) == BG, "background must be black"

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        colors=[BLUE, RED, CRACK],
    )
    assert 1 <= len(objects) <= 2, "expected 1 or 2 crack-marked objects"

    out = np.full_like(g, bg)
    for obj in objects:
        r0, c0, _r1, _c1 = map(int, obj["bbox"])
        local = crop_bbox(g, obj["bbox"])
        transformed, placements = _transform_local_object(local)

        assert r0 + transformed.shape[0] <= g.shape[0]
        assert c0 + transformed.shape[1] <= g.shape[1]

        for patch, pr0, pc0 in placements:
            rr = r0 + pr0
            cc = c0 + pc0
            ph, pw = patch.shape
            mask = patch != bg
            assert not np.any((out[rr : rr + ph, cc : cc + pw] != bg) & mask), "objects must not collide after transformation"
            paste(out, patch, rr, cc, transparent=bg)

    return out


def _random_profile(length: int, span: int) -> np.ndarray:
    assert span >= 2
    values = np.array([random.randint(1, span) for _ in range(length)], dtype=int)
    values[random.randrange(length)] = span
    if np.all(values == span):
        values[random.randrange(length)] = random.randint(1, span - 1)
    return values


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    phase = random.randrange(2)
    coords = np.argwhere(mask)
    for r, c in coords:
        patch[r, c] = BLUE if (int(r) + int(c) + phase) % 2 == 0 else RED

    values = patch[mask]
    if not (np.any(values == BLUE) and np.any(values == RED)):
        first_r, first_c = map(int, coords[0])
        last_r, last_c = map(int, coords[-1])
        patch[first_r, first_c] = BLUE
        patch[last_r, last_c] = RED
    return patch


def _build_wide_object() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        height = random.randint(4, 6)
        left_span = random.randint(2, 4)
        right_span = random.randint(2, 4)
        width = left_span + 1 + right_span
        if width <= height:
            continue

        left_profile = _random_profile(height, left_span)
        right_profile = _random_profile(height, right_span)
        color_area = int(left_profile.sum() + right_profile.sum())
        if not 10 <= color_area <= 18:
            continue

        left_mask = np.zeros((height, left_span), dtype=bool)
        right_mask = np.zeros((height, right_span), dtype=bool)
        for r in range(height):
            left_mask[r, left_span - left_profile[r] : left_span] = True
            right_mask[r, : right_profile[r]] = True

        if np.array_equal(right_mask, np.fliplr(right_mask)):
            continue

        left_patch = _colorize_mask(left_mask)
        right_patch = _colorize_mask(right_mask)

        inp = np.full((height, width), BG, dtype=int)
        inp[:, :left_span] = left_patch
        inp[:, left_span] = CRACK
        inp[:, left_span + 1 :] = right_patch

        out = np.full((2 * height, left_span + right_span), BG, dtype=int)
        out[:height, :left_span] = _swap_blue_red(left_patch)
        out[height:, left_span:] = _swap_blue_red(np.fliplr(right_patch))
        return inp, out
    return None


def _build_object(orientation: str) -> tuple[np.ndarray, np.ndarray] | None:
    base = _build_wide_object()
    if base is None:
        return None
    inp, out = base
    if orientation == "vertical":
        return inp, out
    if orientation == "horizontal":
        return inp.T.copy(), out.T.copy()
    raise ValueError(f"unknown orientation: {orientation}")


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, object_count: int) -> bool:
    input_objects = find_components(inp, bg=BG, connectivity=4, monochromatic=False, colors=[BLUE, RED, CRACK])
    if len(input_objects) != object_count:
        return False
    for obj in input_objects:
        local = crop_bbox(inp, obj["bbox"])
        try:
            orientation, erased, half_a, half_b = _split_object(local)
        except AssertionError:
            return False
        if orientation not in {"vertical", "horizontal"}:
            return False
        b_patch = crop_bbox(erased, half_b["bbox"])
        mirrored = np.fliplr(b_patch) if orientation == "vertical" else np.flipud(b_patch)
        if np.array_equal(b_patch != BG, mirrored != BG):
            return False
        for half in (half_a, half_b):
            values = erased[half["coords"][:, 0], half["coords"][:, 1]]
            if not (np.any(values == BLUE) and np.any(values == RED)):
                return False

    if np.any(expected == CRACK):
        return False
    output_parts = find_components(expected, bg=BG, connectivity=4, monochromatic=False, colors=[BLUE, RED])
    return len(output_parts) == 2 * object_count


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        object_count = 1 if random.random() < 0.55 else 2
        if object_count == 1:
            rows = random.randint(14, 18)
            cols = random.randint(14, 18)
        else:
            rows = random.randint(18, 24)
            cols = random.randint(18, 24)

        orientation_choices = [random.choice(["vertical", "horizontal"]) for _ in range(object_count)]
        if object_count == 2 and random.random() < 0.5:
            orientation_choices = ["vertical", "horizontal"]
            random.shuffle(orientation_choices)

        specs: list[tuple[np.ndarray, np.ndarray]] = []
        for orientation in orientation_choices:
            built = _build_object(orientation)
            if built is None:
                break
            specs.append(built)
        if len(specs) != object_count:
            continue

        specs.sort(key=lambda pair: pair[1].shape[0] * pair[1].shape[1], reverse=True)
        placements: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        input_boxes: list[tuple[int, int, int, int]] = []
        output_boxes: list[tuple[int, int, int, int]] = []

        failed = False
        for inp_patch, out_patch in specs:
            ih, iw = inp_patch.shape
            oh, ow = out_patch.shape
            need_h = max(ih, oh)
            need_w = max(iw, ow)
            if need_h > rows or need_w > cols:
                failed = True
                break

            candidates = [(r, c) for r in range(rows - need_h + 1) for c in range(cols - need_w + 1)]
            random.shuffle(candidates)

            chosen = None
            for r0, c0 in candidates:
                in_box = (r0, c0, r0 + ih - 1, c0 + iw - 1)
                out_box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                if any(not _boxes_separated(in_box, old, pad=1) for old in input_boxes):
                    continue
                if any(not _boxes_separated(out_box, old, pad=1) for old in output_boxes):
                    continue
                chosen = (r0, c0)
                break

            if chosen is None:
                failed = True
                break

            r0, c0 = chosen
            placements.append((inp_patch, out_patch, r0, c0))
            input_boxes.append((r0, c0, r0 + ih - 1, c0 + iw - 1))
            output_boxes.append((r0, c0, r0 + oh - 1, c0 + ow - 1))

        if failed:
            continue

        inp = np.full((rows, cols), BG, dtype=int)
        expected = np.full((rows, cols), BG, dtype=int)
        for inp_patch, out_patch, r0, c0 in placements:
            paste(inp, inp_patch, r0, c0, transparent=BG)
            paste(expected, out_patch, r0, c0, transparent=BG)

        if not _semantic_ok(inp, expected, object_count):
            continue

        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid crack-split regrouping input")
