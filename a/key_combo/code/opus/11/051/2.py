from __future__ import annotations

import random
from typing import Sequence

import numpy as np


BG = 0
PALETTE = (2, 3, 7)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _is_connected4(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _is_symmetric(mask: np.ndarray, axis: str) -> bool:
    if axis == "lr":
        return np.array_equal(mask, np.fliplr(mask))
    assert axis == "tb"
    return np.array_equal(mask, np.flipud(mask))


def _repair_patch(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "lr":
        return mask | np.fliplr(mask)
    assert axis == "tb"
    return mask | np.flipud(mask)


def _paint_mask(out: np.ndarray, mask: np.ndarray, color: int) -> None:
    if np.any((out != BG) & mask):
        raise AssertionError("objects must not overlap")
    out[mask] = int(color)


def _recover_object(mask: np.ndarray, forced_axis: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "object must be non-empty"
    bottom = int(coords[:, 0].max())
    recovered = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c + (bottom - r))
        assert 0 <= cc < W, "inverse shear should stay in bounds"
        recovered[rr, cc] = True

    box = _bbox(recovered)
    assert box is not None
    patch = _crop(recovered, box)
    axis = forced_axis or ("lr" if patch.shape[1] > patch.shape[0] else "tb")
    repaired_patch = _repair_patch(patch, axis)
    repaired = np.zeros_like(mask, dtype=bool)
    r0, c0, r1, c1 = box
    repaired[r0 : r1 + 1, c0 : c1 + 1] = repaired_patch
    return recovered, repaired, axis


def _mirror_without_inverse(mask: np.ndarray, axis: str) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    patch = _crop(mask, box)
    repaired = np.zeros_like(mask, dtype=bool)
    r0, c0, r1, c1 = box
    repaired[r0 : r1 + 1, c0 : c1 + 1] = _repair_patch(patch, axis)
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo the left shear on each colored object, then mirror-repair the pixels lost to clipping.

    Input structure:
    - The grid has black background and 2-3 non-touching single-color foreground objects.
    - Each object uses a distinct color and is a left-sheared version of an originally symmetric shape.
    - The bottom row of each object is the shear anchor, while some upper-row pixels may be missing because the shear clipped them at the left edge.

    Rule:
    - Treat each non-background color as one object and identify its bottommost row as the anchor.
    - Invert the shear by moving every pixel in row r right by (bottom - r) columns.
    - Inspect the recovered object's bounding box: if width > height, mirror its existing pixels left-right inside that box; otherwise mirror them top-bottom.
    - Keep the union of the recovered pixels and their reflected copies, preserving the object's color.

    Key insight:
    - The mirror axis cannot be chosen from the sheared input alone: first the shape must be unsheared, and only then does its recovered aspect ratio reveal whether the missing pixels should come from a horizontal or vertical reflection.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    assert 1 <= len(colors) <= 3, "expected 1-3 foreground colors"

    for color in colors:
        mask = g == int(color)
        recovered, repaired, _axis = _recover_object(mask)
        assert np.any(recovered), "inverse shear must keep pixels"
        _paint_mask(out, repaired, color)

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_H_RANGE = (15, 19)
GRID_W_RANGE = (15, 19)
OBJECT_COUNT_CHOICES = (2, 2, 2, 3)
SHAPE_RETRIES = 100


def _wide_patch() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(4, 6)
        w = random.randint(max(5, h + 1), min(9, h + 4))
        max_pad = (w - 1) // 2
        pads = [random.randint(0, max_pad) for _ in range(h)]
        pads[-1] = 0
        if all(p == 0 for p in pads[:-1]):
            pads[random.randrange(h - 1)] = min(max_pad, 1)
        if pads == pads[::-1]:
            continue
        patch = np.zeros((h, w), dtype=bool)
        for r, pad in enumerate(pads):
            patch[r, pad : w - pad] = True
        if not _is_connected4(patch):
            continue
        if not _is_symmetric(patch, "lr"):
            continue
        if _is_symmetric(patch, "tb"):
            continue
        return patch
    raise RuntimeError("failed to build wide patch")


def _row_walk_candidates(prev_l: int, prev_r: int, width: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for nl in range(max(0, prev_l - 1), min(prev_l + 2, width)):
        for nr in range(max(nl, prev_r - 1), min(prev_r + 2, width)):
            if max(prev_l, nl) <= min(prev_r, nr):
                candidates.append((nl, nr))
    return candidates


def _tall_patch() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if random.random() < 0.25:
            w = random.randint(4, 6)
            h = w
        else:
            h = random.randint(5, 8)
            w = random.randint(3, min(6, h))
        half = (h + 1) // 2
        rows = [(0, w - 1)]
        while len(rows) < half:
            prev_l, prev_r = rows[-1]
            candidates = _row_walk_candidates(prev_l, prev_r, w)
            random.shuffle(candidates)
            rows.append(candidates[0])

        mirrored = rows[:-1][::-1] if h % 2 else rows[::-1]
        full_rows = rows + mirrored
        patch = np.zeros((h, w), dtype=bool)
        for r, (lft, rgt) in enumerate(full_rows):
            patch[r, lft : rgt + 1] = True

        if not _is_connected4(patch):
            continue
        if not _is_symmetric(patch, "tb"):
            continue
        if _is_symmetric(patch, "lr"):
            continue
        if not any(lft != (w - 1 - rgt) for lft, rgt in full_rows):
            continue
        return patch
    raise RuntimeError("failed to build tall patch")


def _place_patch_mask(shape: tuple[int, int], patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    H, W = shape
    h, w = patch.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = patch
    return mask


def _shear_left(mask: np.ndarray) -> tuple[np.ndarray, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    bottom = int(coords[:, 0].max())
    sheared = np.zeros_like(mask, dtype=bool)
    lost = 0
    for r, c in coords:
        rr = int(r)
        cc = int(c - (bottom - r))
        if 0 <= cc < mask.shape[1]:
            sheared[rr, cc] = True
        else:
            lost += 1
    return sheared, lost


def _object_candidate(
    patch: np.ndarray,
    axis: str,
    color: int,
    shape: tuple[int, int],
    r0: int,
    c0: int,
) -> dict | None:
    area = int(patch.sum())
    out_mask = _place_patch_mask(shape, patch, r0, c0)
    in_mask, lost = _shear_left(out_mask)
    if lost <= 0 or 2 * lost > area:
        return None

    recovered, repaired, inferred_axis = _recover_object(in_mask)
    wrong_axis = "tb" if axis == "lr" else "lr"
    _wrong_recovered, wrong_repaired, _ = _recover_object(in_mask, forced_axis=wrong_axis)
    direct_repaired = _mirror_without_inverse(in_mask, axis)

    if inferred_axis != axis:
        return None
    if np.array_equal(recovered, out_mask):
        return None
    if not np.array_equal(repaired, out_mask):
        return None
    if np.array_equal(wrong_repaired, out_mask):
        return None
    if np.array_equal(direct_repaired, out_mask):
        return None

    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    inp[in_mask] = color
    out[out_mask] = color
    if not np.array_equal(main(inp), out):
        return None

    return {
        "color": int(color),
        "axis": axis,
        "in_mask": in_mask,
        "out_mask": out_mask,
        "lost": lost,
    }


def _enumerate_candidates(patch: np.ndarray, axis: str, color: int, shape: tuple[int, int]) -> list[dict]:
    H, W = shape
    h, w = patch.shape
    candidates: list[dict] = []
    r_values = list(range(max(0, H - h + 1)))
    c_values = list(range(max(0, W - w + 1)))
    random.shuffle(r_values)
    random.shuffle(c_values)
    for r0 in r_values:
        for c0 in c_values:
            cand = _object_candidate(patch, axis, color, shape, r0, c0)
            if cand is not None:
                candidates.append(cand)
    random.shuffle(candidates)
    return candidates


def _non_touching(existing: np.ndarray, candidate: np.ndarray) -> bool:
    return not np.any(_dilate8(existing) & candidate)


def _backtrack_place(candidates_per_object: Sequence[list[dict]]) -> list[dict] | None:
    n = len(candidates_per_object)
    order = sorted(range(n), key=lambda i: len(candidates_per_object[i]))
    H, W = candidates_per_object[order[0]][0]["in_mask"].shape
    chosen: list[dict | None] = [None] * n
    in_occ = np.zeros((H, W), dtype=bool)
    out_occ = np.zeros((H, W), dtype=bool)

    def rec(idx: int) -> bool:
        nonlocal in_occ, out_occ
        if idx == n:
            return True
        obj_idx = order[idx]
        for cand in candidates_per_object[obj_idx]:
            if not _non_touching(in_occ, cand["in_mask"]):
                continue
            if not _non_touching(out_occ, cand["out_mask"]):
                continue
            prev_in = in_occ.copy()
            prev_out = out_occ.copy()
            in_occ |= cand["in_mask"]
            out_occ |= cand["out_mask"]
            chosen[obj_idx] = cand
            if rec(idx + 1):
                return True
            in_occ = prev_in
            out_occ = prev_out
            chosen[obj_idx] = None
        return False

    if not rec(0):
        return None
    return [cand for cand in chosen if cand is not None]


def _scene_from_choices(shape: tuple[int, int], choices: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for cand in choices:
        _paint_mask(inp, cand["in_mask"], cand["color"])
        _paint_mask(out, cand["out_mask"], cand["color"])
    return inp, out


def _solve_scene_with_forced_axis(grid: np.ndarray, forced_axis: str | None = None, inverse_only: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        mask = g == color
        recovered, repaired, _axis = _recover_object(mask, forced_axis=forced_axis)
        chosen = recovered if inverse_only else repaired
        if np.any((out != BG) & chosen):
            raise AssertionError("forced alternate solve overlaps objects")
        out[chosen] = color
    return out


def _validate_scene(inp: np.ndarray, out: np.ndarray, choices: Sequence[dict]) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    try:
        if np.array_equal(_solve_scene_with_forced_axis(inp, inverse_only=True), out):
            return False
    except AssertionError:
        pass
    try:
        if np.array_equal(_solve_scene_with_forced_axis(inp, forced_axis="lr"), out):
            return False
    except AssertionError:
        pass
    try:
        if np.array_equal(_solve_scene_with_forced_axis(inp, forced_axis="tb"), out):
            return False
    except AssertionError:
        pass
    branches = {cand["axis"] for cand in choices}
    if branches != {"lr", "tb"}:
        return False
    if any(int(cand["lost"]) <= 0 for cand in choices):
        return False
    return True


def _object_specs() -> list[tuple[np.ndarray, str]]:
    specs: list[tuple[np.ndarray, str]] = [(_wide_patch(), "lr"), (_tall_patch(), "tb")]
    if random.choice(OBJECT_COUNT_CHOICES) == 3:
        specs.append((_wide_patch(), "lr") if random.random() < 0.5 else (_tall_patch(), "tb"))
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    specs = _object_specs()
    max_h = max(int(p.shape[0]) for p, _axis in specs)
    max_w = max(int(p.shape[1]) for p, _axis in specs)
    min_h = max(GRID_H_RANGE[0], max_h + 2)
    min_w = max(GRID_W_RANGE[0], max_w + 2)
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    H = random.randint(min_h, GRID_H_RANGE[1] - 1)
    W = random.randint(min_w, GRID_W_RANGE[1] - 1)
    shape = (H, W)
    colors = random.sample(PALETTE, len(specs))

    candidates_per_object: list[list[dict]] = []
    for (patch, axis), color in zip(specs, colors):
        candidates = _enumerate_candidates(patch, axis, color, shape)
        if not candidates:
            return None
        candidates_per_object.append(candidates[:80])

    choices = _backtrack_place(candidates_per_object)
    if choices is None:
        return None

    inp, out = _scene_from_choices(shape, choices)
    if not _validate_scene(inp, out, choices):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
