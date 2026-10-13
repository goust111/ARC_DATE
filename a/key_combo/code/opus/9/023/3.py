from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox


BLACK = Color.BLACK
YELLOW = Color.YELLOW
PINK = Color.PINK

MAX_ATTEMPTS = 600


def _cells_of_color(arr: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(arr == int(color))]


def _all_orientations(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(arr, k)
        variants.append(rotated.copy())
        variants.append(np.fliplr(rotated).copy())
    return variants


def _pink_signature(arr: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset(_cells_of_color(arr, PINK))


def _extract_template(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    assert W % 2 == 0, "expected an even grid width"
    split = W // 2
    assert split % 2 == 0, "expected the left half width to be even"

    left = grid[:, :split]
    assert np.all(left[1::2, :] == BLACK), "odd rows in the left zone must be separators"
    assert np.all(left[:, 1::2] == BLACK), "odd columns in the left zone must be separators"

    compact = left[::2, ::2]
    bb = bbox(compact != BLACK)
    assert bb is not None, "expected a non-empty encoded template"
    template = crop_bbox(compact, bb)
    assert np.all(np.isin(template, [YELLOW, PINK])), "template must use only yellow and magenta"
    assert np.all(template != BLACK), "template must tile its bounding box without holes"
    assert (template == YELLOW).any() and (template == PINK).any(), "template must contain both colors"
    return template


def _detect_placements(
    right: np.ndarray, variants: list[np.ndarray]
) -> set[tuple[int, int, int]]:
    H, W = right.shape
    anchors = [tuple(map(int, rc)) for rc in np.argwhere(right == PINK)]
    placements: set[tuple[int, int, int]] = set()

    for anchor_r, anchor_c in anchors:
        for vid, variant in enumerate(variants):
            pink_cells = _cells_of_color(variant, PINK)
            for ref_r, ref_c in pink_cells:
                top = anchor_r - ref_r
                left = anchor_c - ref_c
                if all(
                    0 <= top + r < H
                    and 0 <= left + c < W
                    and int(right[top + r, left + c]) == PINK
                    for r, c in pink_cells
                ):
                    placements.add((vid, int(top), int(left)))
    return placements


def _render_from_placements(
    grid: np.ndarray, variants: list[np.ndarray], placements: set[tuple[int, int, int]]
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    split = out.shape[1] // 2
    right_w = out.shape[1] - split

    for vid, top, left in sorted(placements):
        for r, c in _cells_of_color(variants[vid], YELLOW):
            rr = top + r
            cc = left + c
            if 0 <= rr < out.shape[0] and 0 <= cc < right_w and int(out[rr, split + cc]) == BLACK:
                out[rr, split + cc] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the sparse left lattice into a hidden template, then use its magenta signature to complete right-side copies.

    Input structure:
    - The grid has a black background and is split vertically into an even-width left zone and a right zone.
    - In the left zone, every odd-indexed row and odd-indexed column is black; the even/even crossings encode a compact yellow/magenta template.
    - In the right zone, several magenta pixel clusters are the complete magenta signatures of rotated or reflected copies of that hidden template, while the corresponding yellow pixels are missing.

    Rule:
    - Downsample the left zone by keeping only even rows and even columns, then crop the non-black bounding box to recover the compact template.
    - Generate all 8 dihedral orientations of the template. For every right-zone magenta pixel used as an anchor, align every magenta cell of every orientation to that anchor; if all magenta cells of that oriented signature land in-bounds on magenta pixels, the placement is valid.
    - For each valid placement, paint every in-bounds yellow cell of that orientation yellow if the destination cell is currently black. Leave magenta cells unchanged and never modify the left zone.

    Key insight:
    - The true template is not directly visible: it must first be decoded from the sparse even/even lattice, and only the magenta subset of each oriented copy is present on the right as an alignment signature for restoring the missing yellow cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    template = _extract_template(g)
    variants = _all_orientations(template)
    signatures = [_pink_signature(variant) for variant in variants]
    assert len(set(signatures)) == 8, "magenta signatures must uniquely determine the orientation"

    split = g.shape[1] // 2
    placements = _detect_placements(g[:, split:], variants)
    return _render_from_placements(g, variants, placements)


def _template_fits(compact_h: int, compact_w: int, h: int, w: int) -> bool:
    return 3 <= h <= compact_h and 3 <= w <= compact_w


def _random_template(
    compact_h: int, compact_w: int, *, force_two_pinks: bool
) -> np.ndarray | None:
    for _ in range(300):
        h = random.randint(3, min(5, compact_h))
        w = random.randint(3, min(5, compact_w))
        if not _template_fits(compact_h, compact_w, h, w):
            continue
        pink_count = 2 if force_two_pinks else random.randint(2, min(4, h * w - 1))
        flat = random.sample(range(h * w), pink_count)
        template = np.full((h, w), YELLOW, dtype=int)
        for idx in flat:
            template[idx // w, idx % w] = PINK
        if len({_pink_signature(variant) for variant in _all_orientations(template)}) != 8:
            continue
        return template
    return None


def _candidate_records(H: int, right_w: int, variants: list[np.ndarray]) -> list[dict]:
    records: list[dict] = []
    for vid, variant in enumerate(variants):
        pink_cells = _cells_of_color(variant, PINK)
        yellow_cells = _cells_of_color(variant, YELLOW)
        max_pink_r = max(r for r, _c in pink_cells)
        max_pink_c = max(c for _r, c in pink_cells)

        for top in range(-min(r for r, _c in pink_cells), H - max_pink_r):
            for left in range(0, right_w - max_pink_c):
                pink_abs = {(top + r, left + c) for r, c in pink_cells}
                visible_yellow = {
                    (top + r, left + c)
                    for r, c in yellow_cells
                    if 0 <= top + r < H and 0 <= left + c < right_w
                }
                if not visible_yellow:
                    continue
                clipped = len(visible_yellow) < len(yellow_cells)
                records.append(
                    {
                        "key": (vid, int(top), int(left)),
                        "vid": int(vid),
                        "top": int(top),
                        "left": int(left),
                        "pink": pink_abs,
                        "yellow": visible_yellow,
                        "clipped": bool(clipped),
                    }
                )
    random.shuffle(records)
    return records


def _pink_touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _compatible_with_chosen(
    cand: dict,
    chosen: list[dict],
    *,
    allow_share: bool,
) -> bool:
    union_pink = set().union(*(entry["pink"] for entry in chosen)) if chosen else set()
    union_yellow = set().union(*(entry["yellow"] for entry in chosen)) if chosen else set()

    if cand["yellow"] & union_yellow:
        return False
    if cand["yellow"] & union_pink:
        return False
    if cand["pink"] & union_yellow:
        return False

    shared = cand["pink"] & union_pink
    if shared:
        if not allow_share or len(shared) != 1:
            return False
        if _pink_touches(cand["pink"] - shared, union_pink - shared):
            return False
    else:
        if _pink_touches(cand["pink"], union_pink):
            return False
    return True


def _choose_placements(candidates: list[dict], count: int, want_clip: bool, want_share: bool) -> list[dict] | None:
    for _ in range(160):
        chosen: list[dict] = []
        clip_done = not want_clip
        share_done = not want_share

        for _step in range(count):
            union_pink = set().union(*(entry["pink"] for entry in chosen)) if chosen else set()
            needed_share = want_share and not share_done and len(chosen) >= 1
            pool: list[dict] = []

            for cand in candidates:
                if any(cand["key"] == entry["key"] for entry in chosen):
                    continue
                shared_now = bool(cand["pink"] & union_pink)
                if want_share and share_done and shared_now:
                    continue
                if not _compatible_with_chosen(cand, chosen, allow_share=needed_share):
                    continue
                pool.append(cand)

            if needed_share:
                pool = [cand for cand in pool if cand["pink"] & union_pink]
            if want_clip and not clip_done:
                clipped_pool = [cand for cand in pool if cand["clipped"]]
                if clipped_pool:
                    pool = clipped_pool
            if not pool:
                break

            used_vids = {entry["vid"] for entry in chosen}
            fresh_vid_pool = [cand for cand in pool if cand["vid"] not in used_vids]
            if fresh_vid_pool and random.random() < 0.7:
                pool = fresh_vid_pool

            cand = random.choice(pool)
            chosen.append(cand)
            clip_done = clip_done or cand["clipped"]
            share_done = share_done or bool(cand["pink"] & union_pink)

        if len(chosen) == count and clip_done and share_done:
            return chosen
    return None


def _encode_template_left(grid: np.ndarray, template: np.ndarray, split: int) -> tuple[int, int]:
    compact_h = (grid.shape[0] + 1) // 2
    compact_w = split // 2
    max_top = compact_h - template.shape[0]
    max_left = compact_w - template.shape[1]
    top = random.randint(0, max_top)
    left = random.randint(0, max_left)

    for r in range(template.shape[0]):
        for c in range(template.shape[1]):
            grid[2 * (top + r), 2 * (left + c)] = int(template[r, c])
    return top, left


def _paint_expected_right(
    inp: np.ndarray, variants: list[np.ndarray], placements: list[dict]
) -> np.ndarray:
    out = inp.copy()
    split = inp.shape[1] // 2
    for placement in placements:
        vid = placement["vid"]
        top = placement["top"]
        left = placement["left"]
        for r, c in _cells_of_color(variants[vid], YELLOW):
            rr = top + r
            cc = left + c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] - split and int(out[rr, split + cc]) == BLACK:
                out[rr, split + cc] = YELLOW
    return out


def _validate_semantics(
    inp: np.ndarray,
    template: np.ndarray,
    placements: list[dict],
    expected_out: np.ndarray,
) -> bool:
    split = inp.shape[1] // 2
    extracted = _extract_template(inp)
    if not np.array_equal(extracted, template):
        return False

    variants = _all_orientations(extracted)
    if len({_pink_signature(variant) for variant in variants}) != 8:
        return False

    detected = _detect_placements(inp[:, split:], variants)
    planned = {entry["key"] for entry in placements}
    if detected != planned:
        return False

    all_pink = set().union(*(entry["pink"] for entry in placements))
    if {tuple(map(int, rc)) for rc in np.argwhere(inp[:, split:] == PINK)} != all_pink:
        return False

    all_yellow = [entry["yellow"] for entry in placements]
    if any(not cells for cells in all_yellow):
        return False
    if any(all_yellow[i] & all_yellow[j] for i in range(len(all_yellow)) for j in range(i + 1, len(all_yellow))):
        return False

    if np.array_equal(inp, expected_out):
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected_out)


def _attempt() -> np.ndarray | None:
    H = random.randint(11, 17)
    split = random.choice([10, 12, 14])
    W = split * 2

    compact_h = (H + 1) // 2
    compact_w = split // 2

    copy_count = random.choices([1, 2, 3], weights=[1, 4, 4], k=1)[0]
    want_share = copy_count >= 2 and random.random() < 0.35
    want_clip = random.random() < 0.55

    template = _random_template(compact_h, compact_w, force_two_pinks=want_share)
    if template is None:
        return None

    variants = _all_orientations(template)
    candidates = _candidate_records(H, W - split, variants)
    placements = _choose_placements(candidates, copy_count, want_clip, want_share)
    if placements is None:
        return None

    inp = np.zeros((H, W), dtype=int)
    _encode_template_left(inp, template, split)

    for placement in placements:
        for r, c in placement["pink"]:
            inp[r, split + c] = PINK

    expected_out = _paint_expected_right(inp, variants, placements)
    if not _validate_semantics(inp, template, placements, expected_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse template completion input")
