from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
SUBJECT_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OCCLUDER_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
MAX_GENERATION_ATTEMPTS = 300

Rect = tuple[int, int, int, int]


def _mirror_col(axis: float, c: int) -> int:
    mc = 2 * axis - c
    if abs(mc - round(mc)) > 1e-9:
        raise ValueError("mirror column must stay on the integer lattice")
    return int(round(mc))


def _bbox_from_mask(mask: np.ndarray) -> Rect | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _solid_rectangles_of_color(grid: np.ndarray, color: int) -> list[Rect]:
    rects: list[Rect] = []
    for coords in _components(grid == color):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if int(coords.shape[0]) == h * w:
            rects.append((int(r0), int(c0), int(r1), int(c1)))
    rects.sort()
    return rects


def _rect_shape(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _rects_separated(a: Rect, b: Rect, gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _infer_subject_bbox(grid: np.ndarray, occluders: list[Rect]) -> Rect:
    subject = (grid != BLACK) & (grid != GREY) & (grid != TEAL)
    assert subject.any(), "expected a visible subject"
    mask = subject.copy()
    for r0, c0, r1, c1 in occluders:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    bbox = _bbox_from_mask(mask)
    assert bbox is not None, "subject bbox must exist"
    return bbox


def _infer_axis(grid: np.ndarray, occluders: list[Rect]) -> tuple[float, Rect]:
    bbox = _infer_subject_bbox(grid, occluders)
    r0, c0, r1, c1 = bbox
    axis = (c0 + c1) / 2.0
    subject = (grid != BLACK) & (grid != GREY) & (grid != TEAL)

    for r, c in np.argwhere(subject):
        rr = int(r)
        cc = int(c)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            continue
        mc = _mirror_col(axis, cc)
        assert c0 <= mc <= c1, "mirror partner must stay inside the subject bbox"
        partner = int(grid[rr, mc])
        assert partner not in (GREY,), "answer boxes must stay away from the subject"
        if partner not in (BLACK, TEAL):
            assert partner == int(grid[rr, cc]), "visible subject pixels must agree under reflection"

    return axis, bbox


def _reconstruct_subject(grid: np.ndarray, axis: float, bbox: Rect) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.asarray(grid, dtype=int).copy()
    out[out == TEAL] = BLACK
    out[out == GREY] = BLACK

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            value = int(out[r, c])
            if value == BLACK:
                continue
            mc = _mirror_col(axis, c)
            other = int(out[r, mc])
            assert other in (BLACK, value), "reconstruction may only fill missing symmetric cells"
            out[r, mc] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase cyan occluders, mirror-complete the hidden symmetric subject, then transcribe each recovered patch into its matching grey answer box.

    Input structure:
    - A black background contains one multicolor subject with left-right mirror symmetry.
    - Several solid cyan rectangles occlude different parts of that subject.
    - The same number of solid grey rectangles sit away from the subject, and each grey rectangle matches one occluder's exact height and width.

    Rule:
    - Treat cyan rectangles as occluders and grey rectangles as remote answer boxes.
    - Ignore the cyan pixels, infer the subject's vertical mirror axis, erase all cyan rectangles, and reflect the visible subject pixels to reconstruct the hidden content.
    - For each cyan rectangle, extract the fully reconstructed content under that rectangle, including any black cells, and stamp that patch unchanged into the grey rectangle of matching size.

    Key insight:
    - Size matching only tells you where each answer goes; the patch itself can only be known after the occluded subject is rebuilt by mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    occluders = _solid_rectangles_of_color(g, TEAL)
    answers = _solid_rectangles_of_color(g, GREY)
    assert occluders, "expected at least one cyan occluder"
    assert len(occluders) == len(answers), "occluders and answer boxes must pair one-to-one"

    occ_shapes = [_rect_shape(rect) for rect in occluders]
    ans_shapes = [_rect_shape(rect) for rect in answers]
    assert len(set(occ_shapes)) == len(occ_shapes), "occluder sizes must be unique"
    assert sorted(occ_shapes) == sorted(ans_shapes), "grey boxes must match the occluder sizes exactly"

    axis, subject_bbox = _infer_axis(g, occluders)
    out = _reconstruct_subject(g, axis, subject_bbox)

    answer_by_shape = {_rect_shape(rect): rect for rect in answers}
    for rect in occluders:
        r0, c0, r1, c1 = rect
        ar0, ac0, ar1, ac1 = answer_by_shape[_rect_shape(rect)]
        out[ar0 : ar1 + 1, ac0 : ac1 + 1] = out[r0 : r1 + 1, c0 : c1 + 1]

    return out


def _exact_vertical_symmetry(sprite: np.ndarray, axis: float) -> bool:
    h, w = sprite.shape
    for r in range(h):
        for c in range(w):
            mc = 2 * axis - c
            if abs(mc - round(mc)) > 1e-9:
                return False
            mi = int(round(mc))
            if not (0 <= mi < w) or int(sprite[r, c]) != int(sprite[r, mi]):
                return False
    return True


def _make_subject_sprite() -> tuple[np.ndarray, float] | None:
    sh = random.randint(9, 13)
    half_w = random.randint(5, 7)
    has_center = random.choice([False, True])
    sw = 2 * half_w + (1 if has_center else 0)
    axis = (sw - 1) / 2.0

    extents: list[int] = []
    cur = random.randint(max(3, half_w - 2), half_w)
    for _ in range(sh):
        cur = max(3, min(half_w, cur + random.choice([-1, 0, 1])))
        extents.append(cur)
    if len(set(extents)) == 1:
        idx = random.randrange(sh)
        extents[idx] = max(3, min(half_w, extents[idx] + random.choice([-1, 1])))

    mask = np.zeros((sh, sw), dtype=bool)
    for r, extent in enumerate(extents):
        left_start = half_w - extent
        mask[r, left_start:half_w] = True
        if has_center:
            mask[r, half_w] = True
            mask[r, half_w + 1 : half_w + 1 + extent] = True
        else:
            mask[r, half_w : half_w + extent] = True

    for _ in range(random.randint(0, 2)):
        candidates: list[tuple[int, int, int]] = []
        for r in range(1, sh - 1):
            for c in range(1, sw - 1):
                mc = _mirror_col(axis, c)
                if c > mc or abs(c - axis) <= 1.1:
                    continue
                if not mask[r, c] or not mask[r, mc]:
                    continue
                deg_left = sum(mask[r + dr, c + dc] for dr, dc in DIR4)
                deg_right = sum(mask[r + dr, mc + dc] for dr, dc in DIR4)
                if deg_left >= 3 and deg_right >= 3:
                    candidates.append((r, c, mc))
        if candidates:
            r, c, mc = random.choice(candidates)
            mask[r, c] = False
            mask[r, mc] = False

    comps = _components(mask)
    if len(comps) != 1:
        return None

    sprite = np.zeros((sh, sw), dtype=int)
    palette = random.sample(SUBJECT_COLORS, random.randint(2, 4))
    for r in range(sh):
        for c in range(sw):
            if not mask[r, c]:
                continue
            idx = (r + int(abs(c - axis) * 2) + (r // 2)) % len(palette)
            sprite[r, c] = palette[idx]

    if len(np.unique(sprite[sprite != BLACK])) < 2:
        return None
    if not _exact_vertical_symmetry(sprite, axis):
        return None

    for k in range(2 * sw - 1):
        alt_axis = k / 2.0
        if abs(alt_axis - axis) < 1e-9:
            continue
        if _exact_vertical_symmetry(sprite, alt_axis):
            return None

    return sprite, axis


def _candidate_occluders(subject: np.ndarray, axis: float) -> dict[tuple[int, int], list[tuple[int, Rect, Rect, bool]]]:
    sh, sw = subject.shape
    candidates: dict[tuple[int, int], list[tuple[int, Rect, Rect, bool]]] = defaultdict(list)

    for h, w in OCCLUDER_SHAPES:
        for r0 in range(sh - h + 1):
            for c0 in range(sw - w + 1):
                c1 = c0 + w - 1
                if c0 <= axis <= c1:
                    continue
                mc0 = _mirror_col(axis, c1)
                mc1 = _mirror_col(axis, c0)
                if mc0 > mc1:
                    mc0, mc1 = mc1, mc0

                rect = (r0, c0, r0 + h - 1, c1)
                mirror_rect = (r0, mc0, r0 + h - 1, mc1)
                if not _rects_separated(rect, mirror_rect, gap=0):
                    continue

                patch = subject[r0 : r0 + h, c0 : c1 + 1]
                mirror_patch = subject[r0 : r0 + h, mc0 : mc1 + 1]
                if not np.array_equal(patch, mirror_patch):
                    continue

                colored = int(np.count_nonzero(patch))
                if colored < max(3, (h * w) // 2):
                    continue

                mixed_with_black = bool(np.any(patch == BLACK) and np.any(patch != BLACK))
                candidates[(h, w)].append((colored, rect, mirror_rect, mixed_with_black))

    return candidates


def _choose_occluders(subject: np.ndarray, axis: float) -> list[Rect] | None:
    candidates = _candidate_occluders(subject, axis)
    usable_shapes = [shape for shape, options in candidates.items() if options]
    if len(usable_shapes) < 2:
        return None

    target_count = random.choice([2, 3])
    if len(usable_shapes) < target_count:
        target_count = 2

    need_black_mix = random.random() < 0.5

    for _ in range(30):
        shapes = random.sample(usable_shapes, target_count)
        blocked: list[Rect] = []
        chosen: list[Rect] = []
        used_black_mix = False

        def search(i: int) -> bool:
            nonlocal used_black_mix
            if i == len(shapes):
                return (not need_black_mix) or used_black_mix

            options = list(candidates[shapes[i]])
            random.shuffle(options)
            options.sort(key=lambda item: (item[3], item[0]), reverse=True)
            for _, rect, mirror_rect, mixed_with_black in options[:80]:
                if any(not _rects_separated(rect, old, gap=1) or not _rects_separated(mirror_rect, old, gap=1) for old in blocked):
                    continue
                chosen.append(rect)
                blocked.extend([rect, mirror_rect])
                prev = used_black_mix
                used_black_mix = used_black_mix or mixed_with_black
                if search(i + 1):
                    return True
                used_black_mix = prev
                blocked.pop()
                blocked.pop()
                chosen.pop()
            return False

        if search(0):
            return chosen

    return None


def _place_answer_boxes(grid_shape: tuple[int, int], subject_bbox: Rect, shapes: list[tuple[int, int]]) -> list[Rect] | None:
    h, w = grid_shape
    ordered_shapes = sorted(shapes, key=lambda shape: shape[0] * shape[1], reverse=True)
    placed: list[Rect] = []

    def search(i: int) -> bool:
        if i == len(ordered_shapes):
            return True
        rh, rw = ordered_shapes[i]
        candidates: list[Rect] = []
        for r0 in range(h - rh + 1):
            for c0 in range(w - rw + 1):
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if not _rects_separated(rect, subject_bbox, gap=2):
                    continue
                if all(_rects_separated(rect, other, gap=1) for other in placed):
                    candidates.append(rect)
        random.shuffle(candidates)
        for rect in candidates[:200]:
            placed.append(rect)
            if search(i + 1):
                return True
            placed.pop()
        return False

    if not search(0):
        return None

    by_shape = {shape: rect for shape, rect in zip(ordered_shapes, placed)}
    return [by_shape[shape] for shape in shapes]


def _validate_generated_instance(
    inp: np.ndarray,
    out: np.ndarray,
    full_subject: np.ndarray,
    axis: float,
    occluders: list[Rect],
    answers: list[Rect],
) -> bool:
    if len(occluders) != len(answers):
        return False
    if len({_rect_shape(rect) for rect in occluders}) != len(occluders):
        return False
    if sorted(_rect_shape(rect) for rect in occluders) != sorted(_rect_shape(rect) for rect in answers):
        return False

    inferred_axis, inferred_bbox = _infer_axis(inp, occluders)
    if abs(inferred_axis - axis) > 1e-9:
        return False

    subject_bbox = _bbox_from_mask(full_subject != BLACK)
    if subject_bbox != inferred_bbox:
        return False

    for rect in occluders:
        r0, c0, r1, c1 = rect
        patch = full_subject[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero(patch) == 0:
            return False
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if int(full_subject[rr, cc]) == BLACK:
                    continue
                mc = _mirror_col(axis, cc)
                if int(inp[rr, mc]) in (TEAL, GREY):
                    return False
                if int(inp[rr, mc]) != int(full_subject[rr, cc]):
                    return False

    try:
        pred = main(inp)
    except Exception:
        return False
    return np.array_equal(pred, out)


def _gen_attempt() -> np.ndarray | None:
    made = _make_subject_sprite()
    if made is None:
        return None
    subject, axis_local = made
    sh, sw = subject.shape

    h = random.randint(max(24, sh + 8), 30)
    w = random.randint(max(24, sw + 10), 30)
    subject_r0 = random.randint(1, h - sh - 1)
    subject_c0 = random.randint(1, w - sw - 1)
    axis = subject_c0 + axis_local

    occ_local = _choose_occluders(subject, axis_local)
    if occ_local is None:
        return None
    occluders = [
        (subject_r0 + r0, subject_c0 + c0, subject_r0 + r1, subject_c0 + c1)
        for r0, c0, r1, c1 in occ_local
    ]

    subject_bbox = (subject_r0, subject_c0, subject_r0 + sh - 1, subject_c0 + sw - 1)
    shapes = [_rect_shape(rect) for rect in occluders]
    answers = _place_answer_boxes((h, w), subject_bbox, shapes)
    if answers is None:
        return None

    full_subject = np.zeros((h, w), dtype=int)
    full_subject[subject_r0 : subject_r0 + sh, subject_c0 : subject_c0 + sw] = subject

    inp = full_subject.copy()
    out = full_subject.copy()
    for (r0, c0, r1, c1), (ar0, ac0, ar1, ac1) in zip(occluders, answers):
        inp[r0 : r1 + 1, c0 : c1 + 1] = TEAL
        inp[ar0 : ar1 + 1, ac0 : ac1 + 1] = GREY
        out[ar0 : ar1 + 1, ac0 : ac1 + 1] = full_subject[r0 : r1 + 1, c0 : c1 + 1]

    if not _validate_generated_instance(inp, out, full_subject, axis, occluders, answers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-occlusion transcription task")
