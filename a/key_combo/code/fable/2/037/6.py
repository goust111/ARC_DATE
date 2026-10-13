from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int = 0) -> None:
    h, w = patch.shape
    sub = dst[top : top + h, left : left + w]
    mask = patch != transparent
    sub[mask] = patch[mask]
    dst[top : top + h, left : left + w] = sub


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _tight_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    bbox = _bbox_coords(coords)
    return _crop_bbox(mask, bbox).astype(bool), bbox


def _downscale_mask(mask: np.ndarray, factor: int) -> np.ndarray | None:
    h, w = mask.shape
    if factor <= 1 or h % factor != 0 or w % factor != 0:
        return None
    out = np.zeros((h // factor, w // factor), dtype=bool)
    for r in range(0, h, factor):
        for c in range(0, w, factor):
            block = mask[r : r + factor, c : c + factor]
            if not np.all(block == block[0, 0]):
                return None
            out[r // factor, c // factor] = bool(block[0, 0])
    if not out.any():
        return None
    return out


def _extract_object_info(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = _crop_bbox(grid, (r0, c0, r1, c1))
    colors = set(map(int, patch[patch != bg].ravel()))
    assert len(colors) == 1
    color = next(iter(colors))

    obj_mask = patch == color
    bg_comps = _components_mask(~obj_mask, connectivity=4)
    hole_mask = np.zeros_like(obj_mask, dtype=bool)
    inside_count = 0
    for comp in bg_comps:
        on_border = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == obj_mask.shape[0] - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == obj_mask.shape[1] - 1)
        )
        if not on_border:
            inside_count += 1
            hole_mask[comp[:, 0], comp[:, 1]] = True
    assert inside_count == 1, "each hollow object must contain exactly one enclosed hole"

    outer_mask = obj_mask | hole_mask
    hole_crop, hole_bbox = _tight_mask(hole_mask)
    return {
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "patch": patch,
        "obj_mask": obj_mask,
        "hole_mask": hole_mask,
        "hole_crop": hole_crop,
        "hole_bbox": hole_bbox,
        "outer_mask": outer_mask,
    }


def _self_similar_insert(info: dict) -> tuple[np.ndarray, tuple[int, int]] | None:
    outer = info["outer_mask"]
    hole = info["hole_crop"]
    oh, ow = outer.shape
    hh, hw = hole.shape
    if hh == 0 or hw == 0:
        return None
    if oh % hh != 0 or ow % hw != 0:
        return None
    factor_h = oh // hh
    factor_w = ow // hw
    if factor_h != factor_w or factor_h <= 1:
        return None
    factor = factor_h
    if not np.array_equal(_scale_mask(hole, factor), outer):
        return None

    next_mask = _downscale_mask(hole, factor)
    if next_mask is None:
        return None

    hr0, hc0, hr1, hc1 = info["hole_bbox"]
    hole_h = hr1 - hr0 + 1
    hole_w = hc1 - hc0 + 1
    nh, nw = next_mask.shape
    top = hr0 + (hole_h - nh) // 2
    left = hc0 + (hole_w - nw) // 2
    return next_mask, (top, left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend only the hollow objects whose hole is an integer-scaled copy of their own outer silhouette.

    Input structure:
    - The grid has a black background and several separated monochrome hollow objects.
    - Each object encloses exactly one centered background hole.
    - Some holes are the same shape as that object's outer silhouette at a smaller integer scale, while others are different shapes.

    Rule:
    - For each hollow object, reconstruct its outer silhouette by combining the colored ring with its enclosed hole.
    - Compare the hole shape to that object's own outer silhouette using only direct integer scaling, with no rotation or reflection.
    - If they match, place one more smaller solid copy of the same shape at the center of the hole using the object's color; otherwise leave the object unchanged.

    Key insight:
    - The comparison is self-referential: each hole must be matched against the outer shape of the very object that contains it, and only a true self-similar match licenses one more recursive inner layer.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for coords in _components_mask(g != bg, connectivity=4):
        info = _extract_object_info(g, coords, bg)
        insert = _self_similar_insert(info)
        if insert is None:
            continue
        next_mask, (top, left) = insert
        r0, c0, _r1, _c1 = info["bbox"]
        patch = out[r0 : info["bbox"][2] + 1, c0 : info["bbox"][3] + 1]
        sub = patch[top : top + next_mask.shape[0], left : left + next_mask.shape[1]]
        sub[next_mask] = info["color"]
        patch[top : top + next_mask.shape[0], left : left + next_mask.shape[1]] = sub
        out[r0 : info["bbox"][2] + 1, c0 : info["bbox"][3] + 1] = patch
    return out


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_MIN = 22
GRID_MAX = 30
PAD = 2
MAX_ATTEMPTS = 400
BASE_PLUS = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)


def _make_hollow_patch(
    outer_mask: np.ndarray,
    hole_mask_small: np.ndarray,
    color: int,
    *,
    inner_solid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    outer_mask = outer_mask.astype(bool)
    hole_mask_small = hole_mask_small.astype(bool)
    oh, ow = outer_mask.shape
    hh, hw = hole_mask_small.shape
    assert oh > hh and ow > hw
    assert (oh - hh) % 2 == 0 and (ow - hw) % 2 == 0
    top = (oh - hh) // 2
    left = (ow - hw) // 2

    hole_full = np.zeros_like(outer_mask, dtype=bool)
    hole_full[top : top + hh, left : left + hw] = hole_mask_small
    assert np.all(hole_full <= outer_mask), "hole must stay inside outer shape"

    ring = outer_mask & ~hole_full
    assert len(_components_mask(ring, connectivity=4)) == 1

    inp = np.full(outer_mask.shape, BG, dtype=int)
    inp[ring] = color
    out = inp.copy()
    if inner_solid is not None:
        sh, sw = inner_solid.shape
        inner_top = top + (hh - sh) // 2
        inner_left = left + (hw - sw) // 2
        assert np.all(hole_full[inner_top : inner_top + sh, inner_left : inner_left + sw] | (~inner_solid))
        out[inner_top : inner_top + sh, inner_left : inner_left + sw][inner_solid] = color

    info = _extract_object_info(inp, np.argwhere(inp != BG), BG)
    if inner_solid is None:
        assert _self_similar_insert(info) is None
    else:
        insert = _self_similar_insert(info)
        assert insert is not None
        got_mask, _ = insert
        assert np.array_equal(got_mask, inner_solid)
    return inp, out


def _square(side: int) -> np.ndarray:
    return np.ones((side, side), dtype=bool)


def _rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _plus(scale: int) -> np.ndarray:
    return _scale_mask(BASE_PLUS, scale)


def _self_square(color: int) -> tuple[np.ndarray, np.ndarray]:
    next_side, factor = random.choice([(2, 2), (3, 2), (1, 3)])
    inner = _square(next_side)
    hole = _scale_mask(inner, factor)
    outer = _scale_mask(hole, factor)
    return _make_hollow_patch(outer, hole, color, inner_solid=inner)


def _self_rect(color: int) -> tuple[np.ndarray, np.ndarray]:
    inner = _rect(*random.choice([(2, 3), (3, 2)]))
    hole = _scale_mask(inner, 2)
    outer = _scale_mask(hole, 2)
    return _make_hollow_patch(outer, hole, color, inner_solid=inner)


def _self_plus(color: int) -> tuple[np.ndarray, np.ndarray]:
    inner = BASE_PLUS.copy()
    hole = _scale_mask(inner, 2)
    outer = _scale_mask(hole, 2)
    return _make_hollow_patch(outer, hole, color, inner_solid=inner)


def _mismatch_square_plus(color: int) -> tuple[np.ndarray, np.ndarray]:
    outer = _square(12)
    hole = _plus(2)
    return _make_hollow_patch(outer, hole, color, inner_solid=None)


def _mismatch_square_bar(color: int) -> tuple[np.ndarray, np.ndarray]:
    outer_side = random.choice([8, 12])
    outer = _square(outer_side)
    hole = _rect(*random.choice([(2, 6), (6, 2)]))
    return _make_hollow_patch(outer, hole, color, inner_solid=None)


def _mismatch_rect(color: int) -> tuple[np.ndarray, np.ndarray]:
    outer = _rect(*random.choice([(8, 12), (12, 8)]))
    hole = random.choice([_square(4), _plus(2), _rect(2, 6), _rect(6, 2)])
    return _make_hollow_patch(outer, hole, color, inner_solid=None)


SELF_BUILDERS = [_self_square, _self_rect, _self_plus]
MISMATCH_BUILDERS = [_mismatch_square_plus, _mismatch_square_bar, _mismatch_rect]


def _bbox_clear(candidate: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], pad: int) -> bool:
    r0, c0, r1, c1 = candidate
    for br0, bc0, br1, bc1 in boxes:
        row_sep = r1 + pad < br0 or br1 + pad < r0
        col_sep = c1 + pad < bc0 or bc1 + pad < c0
        if not (row_sep or col_sep):
            return False
    return True


def _place_candidates(
    shape: tuple[int, int],
    grid_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    gh, gw = grid_shape
    cands = []
    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _bbox_clear(bbox, boxes, PAD):
                cands.append((r0, c0))
    random.shuffle(cands)
    return cands


def _build_case() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    n_objects = random.choice([2, 3, 3, 4])
    n_self = random.randint(1, n_objects - 1)
    n_mismatch = n_objects - n_self
    colors = random.sample(COLORS, n_objects)

    objects: list[dict] = []
    for i in range(n_self):
        inp_patch, out_patch = random.choice(SELF_BUILDERS)(colors[i])
        objects.append({"inp": inp_patch, "out": out_patch, "self": True})
    for i in range(n_mismatch):
        inp_patch, out_patch = random.choice(MISMATCH_BUILDERS)(colors[n_self + i])
        objects.append({"inp": inp_patch, "out": out_patch, "self": False})
    random.shuffle(objects)

    max_h = max(obj["inp"].shape[0] for obj in objects)
    max_w = max(obj["inp"].shape[1] for obj in objects)
    min_h = max(GRID_MIN, max_h + 2)
    min_w = max(GRID_MIN, max_w + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    gh = random.randint(min_h, GRID_MAX)
    gw = random.randint(min_w, GRID_MAX)

    inp = np.full((gh, gw), BG, dtype=int)
    out = np.full((gh, gw), BG, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    metas: list[dict] = []

    for obj in sorted(objects, key=lambda x: x["inp"].size, reverse=True):
        cands = _place_candidates(obj["inp"].shape, (gh, gw), boxes)
        if not cands:
            return None
        top, left = cands[0]
        _paste(inp, obj["inp"], top, left, transparent=BG)
        _paste(out, obj["out"], top, left, transparent=BG)
        bbox = (top, left, top + obj["inp"].shape[0] - 1, left + obj["inp"].shape[1] - 1)
        boxes.append(bbox)
        metas.append({"bbox": bbox, "self": obj["self"], "inp": obj["inp"], "out": obj["out"]})

    if not any(meta["self"] for meta in metas):
        return None
    if not any(not meta["self"] for meta in metas):
        return None
    if np.array_equal(inp, out):
        return None
    return inp, out, metas


def _semantic_checks(inp: np.ndarray, out: np.ndarray, metas: list[dict]) -> None:
    actual = main(inp)
    assert np.array_equal(actual, out)

    changed_objects = 0
    total_changed = int(np.count_nonzero(inp != out))
    expected_changed = 0
    for meta in metas:
        r0, c0, r1, c1 = meta["bbox"]
        got = actual[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(got, meta["out"])
        if meta["self"]:
            changed_objects += 1
            expected_changed += int(np.count_nonzero(meta["inp"] != meta["out"]))
            assert not np.array_equal(meta["inp"], meta["out"])
        else:
            assert np.array_equal(meta["inp"], meta["out"])
    assert changed_objects >= 1
    assert total_changed == expected_changed

    bg = _get_bg(inp)
    comps = _components_mask(inp != bg, connectivity=4)
    assert len(comps) == len(metas)
    transformed = 0
    untouched = 0
    for coords in comps:
        info = _extract_object_info(inp, coords, bg)
        insert = _self_similar_insert(info)
        r0, c0, r1, c1 = info["bbox"]
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = out[r0 : r1 + 1, c0 : c1 + 1]
        if insert is None:
            untouched += 1
            assert np.array_equal(before, after)
        else:
            transformed += 1
            assert not np.array_equal(before, after)
    assert transformed >= 1 and untouched >= 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, out, metas = built
        try:
            _semantic_checks(inp, out, metas)
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid self-similar recursion puzzle")
