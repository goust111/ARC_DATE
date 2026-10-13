from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite


Grid = np.ndarray
BBox = tuple[int, int, int, int]


_GEN_RETRIES = 500
_GEN_CALL_COUNT = 0
_DISTRACTOR_MODES = (
    "same_large_scaled",
    "same_small_scaled",
    "same_large_noisy",
)
_SMALL_SHAPES = (
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
)
_SCALE_FACTORS = (2, 2, 2, 3, 3)
_ALL_COLORS = tuple(range(10))


def _bbox_with_shape(top: int, left: int, shape: tuple[int, int]) -> BBox:
    return int(top), int(left), int(top + shape[0] - 1), int(left + shape[1] - 1)


def _same_scaled_dims(small_shape: tuple[int, int], large_shape: tuple[int, int]) -> int | None:
    sh, sw = map(int, small_shape)
    lh, lw = map(int, large_shape)
    if sh <= 0 or sw <= 0 or lh <= 0 or lw <= 0:
        return None
    if lh % sh != 0 or lw % sw != 0:
        return None
    k_r = lh // sh
    k_c = lw // sw
    if k_r != k_c or k_r <= 1:
        return None
    return int(k_r)


def _block_profile(block: Grid) -> dict | None:
    colors, counts = np.unique(block, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    lead = int(order[0])
    majority = int(colors[lead])
    majority_count = int(counts[lead])
    if len(order) > 1 and int(counts[int(order[1])]) == majority_count:
        return None
    odd = np.argwhere(block != majority)
    return {
        "majority": majority,
        "pure": bool(odd.size == 0),
        "odd_coords": odd,
        "majority_count": majority_count,
    }


def _scaled_pair_info(small_patch: Grid, large_patch: Grid) -> dict | None:
    k = _same_scaled_dims(tuple(small_patch.shape), tuple(large_patch.shape))
    if k is None:
        return None

    mismatches: list[dict] = []
    impure_count = 0
    pure_wrong_count = 0
    sh, sw = small_patch.shape
    for r in range(sh):
        for c in range(sw):
            block = large_patch[r * k : (r + 1) * k, c * k : (c + 1) * k]
            prof = _block_profile(block)
            if prof is None:
                return None
            majority = int(prof["majority"])
            pure = bool(prof["pure"])
            odd_coords = np.asarray(prof["odd_coords"], dtype=int)
            if not pure and int(odd_coords.shape[0]) != 1:
                return None

            small_color = int(small_patch[r, c])
            mismatch = (not pure) or majority != small_color
            if not mismatch:
                continue

            if pure:
                if majority == small_color:
                    return None
                pure_wrong_count += 1
            else:
                if majority != small_color:
                    return None
                impure_count += 1

            mismatches.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "pure": pure,
                    "majority": majority,
                    "odd_coords": odd_coords,
                }
            )

    if len(mismatches) != 2 or impure_count != 1 or pure_wrong_count != 1:
        return None
    return {"k": int(k), "mismatches": mismatches}


def _pair_candidates(grid: Grid) -> tuple[list[dict], list[tuple[int, int, dict]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    matches: list[tuple[int, int, dict]] = []
    for i, small_comp in enumerate(components):
        small_patch = crop_bbox(g, small_comp["bbox"])
        for j, large_comp in enumerate(components):
            if i == j:
                continue
            info = _scaled_pair_info(small_patch, crop_bbox(g, large_comp["bbox"]))
            if info is not None:
                matches.append((int(i), int(j), info))
    return components, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the tampered small image and its integer-scaled copy by using large-block purity to decide which side is wrong.

    Input structure:
    - The background is the dominant color, and there are exactly three separated non-background objects.
    - One object is a multicolor small image, one is its integer-scaled copy where each source cell becomes a k×k block, and the remaining object is an unrelated distractor.
    - The small image has one recolored foreground cell, and the scaled copy has one single-pixel impurity inside one scaled block at a different logical cell.

    Rule:
    - Identify the unique ordered pair of objects related by integer scaling.
    - Compare the small image cell-by-cell against the corresponding k×k blocks of the large copy, treating a block as suspicious whenever it is not pure.
    - There must be exactly two logical mismatches: one pure block whose color disagrees with the small image, and one impure block whose majority color agrees with the small image.
    - Fix the pure-block mismatch on the small image, and fix the impure-block mismatch by repainting the odd pixel in the large copy to that block's majority color.

    Key insight:
    - A disagreement alone does not identify the culprit; only the internal self-consistency of each enlarged k×k block decides whether the small cell or the large block was tampered with.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    components, matches = _pair_candidates(g)
    assert len(components) == 3, "expected exactly three foreground objects"
    assert len(matches) == 1, "expected a unique cross-scale object pair"

    small_index, large_index, info = matches[0]
    small_bbox = components[small_index]["bbox"]
    large_bbox = components[large_index]["bbox"]
    sr0, sc0, _sr1, _sc1 = small_bbox
    lr0, lc0, _lr1, _lc1 = large_bbox
    k = int(info["k"])

    out = g.copy()
    for mismatch in info["mismatches"]:
        r = int(mismatch["r"])
        c = int(mismatch["c"])
        majority = int(mismatch["majority"])
        if bool(mismatch["pure"]):
            out[sr0 + r, sc0 + c] = majority
            continue

        block = out[lr0 + r * k : lr0 + (r + 1) * k, lc0 + c * k : lc0 + (c + 1) * k]
        block[block != majority] = majority

    return out


def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _palette_without(bg: int) -> np.ndarray:
    return np.array([color for color in _ALL_COLORS if int(color) != int(bg)], dtype=int)


def _neighbor_cells(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _connected_mask(rng: np.random.Generator, shape: tuple[int, int]) -> Grid | None:
    h, w = map(int, shape)
    lo = max(h + w - 1, int(np.ceil(0.55 * h * w)))
    hi = h * w
    for _ in range(120):
        target = int(rng.integers(lo, hi + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        frontier = set(_neighbor_cells(start[0], start[1], h, w))
        while len(cells) < target and frontier:
            rr, cc = _pick(rng, tuple(frontier))
            frontier.remove((rr, cc))
            if (rr, cc) in cells:
                continue
            cells.add((rr, cc))
            for nxt in _neighbor_cells(rr, cc, h, w):
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if int(mask.sum()) == h * w and h * w >= 12 and rng.random() < 0.75:
            continue
        return mask
    return None


def _colored_patch(rng: np.random.Generator, shape: tuple[int, int], bg: int) -> Grid | None:
    mask = _connected_mask(rng, shape)
    if mask is None:
        return None

    patch = np.full(shape, int(bg), dtype=int)
    coords = np.argwhere(mask)
    color_count = int(rng.integers(3, min(6, int(coords.shape[0])) + 1))
    palette = _palette_without(bg)
    chosen = list(map(int, rng.choice(palette, size=color_count, replace=False)))
    values = chosen[:]
    while len(values) < int(coords.shape[0]):
        values.append(int(_pick(rng, chosen)))
    rng.shuffle(values)
    for (rr, cc), color in zip(coords, values):
        patch[int(rr), int(cc)] = int(color)
    return patch


def _different_patch(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bg: int,
    reference: Grid,
    min_diff: int = 3,
) -> Grid | None:
    for _ in range(120):
        patch = _colored_patch(rng, shape, bg)
        if patch is None:
            continue
        if int(np.count_nonzero(patch != reference)) >= int(min_diff):
            return patch
    return None


def _occupied_cells(patch: Grid, bg: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != int(bg))]


def _recolor_cell(rng: np.random.Generator, color: int, bg: int) -> int:
    choices = [candidate for candidate in _ALL_COLORS if int(candidate) not in {int(bg), int(color)}]
    return int(_pick(rng, choices))


def _corrupt_small(
    rng: np.random.Generator,
    patch: Grid,
    bg: int,
) -> tuple[Grid, tuple[int, int], int]:
    coords = _occupied_cells(patch, bg)
    rc = _pick(rng, coords)
    rr, cc = map(int, rc)
    out = patch.copy()
    original = int(out[rr, cc])
    out[rr, cc] = _recolor_cell(rng, original, bg)
    return out, (rr, cc), original


def _corrupt_large(
    rng: np.random.Generator,
    patch: Grid,
    bg: int,
    k: int,
    source_patch: Grid,
    forbidden_cell: tuple[int, int],
) -> tuple[Grid, tuple[int, int], tuple[int, int], int]:
    logical_cells = [cell for cell in _occupied_cells(source_patch, bg) if cell != forbidden_cell]
    cell = _pick(rng, logical_cells)
    rr, cc = map(int, cell)
    pr = int(rng.integers(k))
    pc = int(rng.integers(k))
    out = patch.copy()
    ar = rr * k + pr
    ac = cc * k + pc
    original = int(out[ar, ac])
    out[ar, ac] = _recolor_cell(rng, original, bg)
    return out, (rr, cc), (ar, ac), original


def _rect_conflict(a: BBox, b: BBox, padding: int = 1) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_patches(
    rng: np.random.Generator,
    shape: tuple[int, int],
    patches: dict[str, Grid],
) -> dict[str, tuple[int, int]] | None:
    H, W = map(int, shape)
    order = list(patches)
    rng.shuffle(order)
    positions: dict[str, tuple[int, int]] = {}
    placed: list[BBox] = []

    for role in order:
        patch = patches[role]
        ph, pw = map(int, patch.shape)
        r_start = 1 if H - ph >= 2 else 0
        r_stop = H - ph - (1 if H - ph >= 2 else 0)
        c_start = 1 if W - pw >= 2 else 0
        c_stop = W - pw - (1 if W - pw >= 2 else 0)
        candidates = []
        for r in range(r_start, r_stop + 1):
            for c in range(c_start, c_stop + 1):
                bbox = _bbox_with_shape(r, c, patch.shape)
                if any(_rect_conflict(bbox, other, padding=1) for other in placed):
                    continue
                candidates.append((int(r), int(c)))
        if not candidates:
            return None
        pos = _pick(rng, candidates)
        positions[role] = (int(pos[0]), int(pos[1]))
        placed.append(_bbox_with_shape(pos[0], pos[1], patch.shape))
    return positions


def _compose(shape: tuple[int, int], bg: int, patches: dict[str, Grid], positions: dict[str, tuple[int, int]]) -> Grid:
    canvas = np.full(shape, int(bg), dtype=int)
    for role, patch in patches.items():
        r0, c0 = positions[role]
        paste(canvas, patch, r0, c0, transparent=bg)
    return canvas


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    small_clean: Grid,
    large_clean: Grid,
    distractor: Grid,
    positions: dict[str, tuple[int, int]],
) -> Grid:
    return _compose(
        shape,
        bg,
        {"small": small_clean, "large": large_clean, "distractor": distractor},
        positions,
    )


def _false_ratio_pair_count(grid: Grid, matched_pair: tuple[int, int]) -> int:
    components, _matches = _pair_candidates(grid)
    count = 0
    for i, comp_a in enumerate(components):
        patch_a = crop_bbox(grid, comp_a["bbox"])
        for j, comp_b in enumerate(components):
            if i == j or (i, j) == matched_pair:
                continue
            patch_b = crop_bbox(grid, comp_b["bbox"])
            if _same_scaled_dims(tuple(patch_a.shape), tuple(patch_b.shape)) is None:
                continue
            if _scaled_pair_info(patch_a, patch_b) is None:
                count += 1
    return int(count)


def _semantic_ok(
    inp: Grid,
    expected: Grid,
    *,
    small_bbox: BBox,
    large_bbox: BBox,
    small_bad_cell: tuple[int, int],
    large_bad_cell: tuple[int, int],
    k: int,
) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(out, expected):
        return False

    bg = int(get_bg(inp))
    if int(np.count_nonzero(inp == bg)) <= inp.size // 2:
        return False

    components, matches = _pair_candidates(inp)
    if len(components) != 3 or len(matches) != 1:
        return False

    small_index, large_index, info = matches[0]
    if tuple(map(int, components[small_index]["bbox"])) != tuple(map(int, small_bbox)):
        return False
    if tuple(map(int, components[large_index]["bbox"])) != tuple(map(int, large_bbox)):
        return False
    if int(info["k"]) != int(k):
        return False

    pure_positions = {(int(m["r"]), int(m["c"])) for m in info["mismatches"] if bool(m["pure"])}
    impure_positions = {(int(m["r"]), int(m["c"])) for m in info["mismatches"] if not bool(m["pure"])}
    if pure_positions != {tuple(map(int, small_bad_cell))}:
        return False
    if impure_positions != {tuple(map(int, large_bad_cell))}:
        return False

    if _false_ratio_pair_count(inp, (small_index, large_index)) < 1:
        return False
    return True


def _canvas_shape(
    rng: np.random.Generator,
    patches: dict[str, Grid],
    bg: int,
) -> tuple[int, int] | None:
    max_h = max(int(p.shape[0]) for p in patches.values())
    max_w = max(int(p.shape[1]) for p in patches.values())
    fg_total = sum(int(np.count_nonzero(p != int(bg))) for p in patches.values())
    min_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(2 * fg_total + 40))))
    if min_side > 30:
        return None

    for _ in range(80):
        H = int(rng.integers(min_side, 31))
        W = int(rng.integers(min_side, 31))
        if H * W <= 2 * fg_total:
            continue
        return H, W
    return None


def _distractor_patch(
    rng: np.random.Generator,
    mode: str,
    bg: int,
    small_clean: Grid,
    large_clean: Grid,
    k: int,
) -> Grid | None:
    if mode == "same_large_scaled":
        other_small = _different_patch(rng, tuple(small_clean.shape), bg, small_clean, min_diff=3)
        if other_small is None:
            return None
        return scale_sprite(other_small, k)

    if mode == "same_small_scaled":
        return _different_patch(rng, tuple(small_clean.shape), bg, small_clean, min_diff=3)

    if mode == "same_large_noisy":
        patch = _different_patch(rng, tuple(large_clean.shape), bg, large_clean, min_diff=6)
        if patch is None:
            return None
        return patch

    raise ValueError(f"unknown distractor mode: {mode}")


def _gen_attempt(rng: np.random.Generator, mode: str) -> Grid | None:
    bg = int(_pick(rng, _ALL_COLORS))
    small_shape = tuple(map(int, _pick(rng, _SMALL_SHAPES)))
    k = int(_pick(rng, _SCALE_FACTORS))

    small_clean = _colored_patch(rng, small_shape, bg)
    if small_clean is None:
        return None
    large_clean = scale_sprite(small_clean, k)
    distractor = _distractor_patch(rng, mode, bg, small_clean, large_clean, k)
    if distractor is None:
        return None

    small_in, small_bad_cell, _small_original = _corrupt_small(rng, small_clean, bg)
    large_in, large_bad_cell, _pixel_bad_pos, _large_original = _corrupt_large(
        rng, large_clean, bg, k, small_clean, small_bad_cell
    )

    patches_in = {"small": small_in, "large": large_in, "distractor": distractor}
    shape = _canvas_shape(rng, patches_in, bg)
    if shape is None:
        return None
    positions = _place_patches(rng, shape, patches_in)
    if positions is None:
        return None

    inp = _compose(shape, bg, patches_in, positions)
    expected = _expected_output(shape, bg, small_clean, large_clean, distractor, positions)
    small_bbox = _bbox_with_shape(*positions["small"], small_in.shape)
    large_bbox = _bbox_with_shape(*positions["large"], large_in.shape)
    if not _semantic_ok(
        inp,
        expected,
        small_bbox=small_bbox,
        large_bbox=large_bbox,
        small_bad_cell=small_bad_cell,
        large_bad_cell=large_bad_cell,
        k=k,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _DISTRACTOR_MODES[_GEN_CALL_COUNT % len(_DISTRACTOR_MODES)]
    _GEN_CALL_COUNT += 1

    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-scale repair input")
