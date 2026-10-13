from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID = 30
GEN_ATTEMPTS = 400

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _piece_info(patch: np.ndarray, coords: np.ndarray) -> dict:
    colors = [int(v) for v in np.unique(patch) if int(v) != BLACK]
    assert len(colors) == 1, "each piece must be monochrome"
    color = int(colors[0])
    mask = patch == color
    h, w = mask.shape

    row_good = True
    row_mins: list[int] = []
    row_maxs: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0 or int(cols[-1] - cols[0] + 1) != int(cols.size):
            row_good = False
            break
        row_mins.append(int(cols[0]))
        row_maxs.append(int(cols[-1]))

    col_good = True
    col_mins: list[int] = []
    col_maxs: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0 or int(rows[-1] - rows[0] + 1) != int(rows.size):
            col_good = False
            break
        col_mins.append(int(rows[0]))
        col_maxs.append(int(rows[-1]))

    candidates: list[tuple[str, tuple[int, ...]]] = []
    if row_good and all(v == 0 for v in row_mins) and len(set(row_maxs)) > 1:
        candidates.append(("E", tuple(int(w - 1 - v) for v in row_maxs)))
    if row_good and all(v == w - 1 for v in row_maxs) and len(set(row_mins)) > 1:
        candidates.append(("W", tuple(int(v) for v in row_mins)))
    if col_good and all(v == 0 for v in col_mins) and len(set(col_maxs)) > 1:
        candidates.append(("S", tuple(int(h - 1 - v) for v in col_maxs)))
    if col_good and all(v == h - 1 for v in col_maxs) and len(set(col_mins)) > 1:
        candidates.append(("N", tuple(int(v) for v in col_mins)))

    assert len(candidates) == 1, "expected exactly one toothed side"
    tooth_side, profile = candidates[0]

    return {
        "color": color,
        "patch": np.asarray(patch, dtype=int),
        "mask": np.asarray(mask, dtype=bool),
        "bbox": _bbox(coords),
        "coords": np.asarray(coords, dtype=int),
        "tooth_side": tooth_side,
        "profile": tuple(int(v) for v in profile),
        "shape": tuple(int(v) for v in patch.shape),
    }


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]
    assert len(cols) == 1, "expected exactly one grey separator column"
    return int(cols[0])


def _profiles_complement(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    if len(a) != len(b) or not a:
        return False
    sums = [int(x) + int(y) for x, y in zip(a, b)]
    return len(set(sums)) == 1 and sums[0] > 0


def _candidate_specs(pieces: list[dict]) -> list[dict]:
    specs: list[dict] = []
    n = len(pieces)
    for i in range(n):
        for j in range(i + 1, n):
            a = pieces[i]
            b = pieces[j]
            if a["tooth_side"] == "E" and b["tooth_side"] == "W":
                if a["shape"][0] == b["shape"][0] and _profiles_complement(a["profile"], b["profile"]):
                    specs.append({"members": (i, j), "kind": "V", "left": i, "right": j})
            if a["tooth_side"] == "W" and b["tooth_side"] == "E":
                if a["shape"][0] == b["shape"][0] and _profiles_complement(a["profile"], b["profile"]):
                    specs.append({"members": (i, j), "kind": "V", "left": j, "right": i})
            if a["tooth_side"] == "S" and b["tooth_side"] == "N":
                if a["shape"][1] == b["shape"][1] and _profiles_complement(a["profile"], b["profile"]):
                    specs.append({"members": (i, j), "kind": "H", "top": i, "bottom": j})
            if a["tooth_side"] == "N" and b["tooth_side"] == "S":
                if a["shape"][1] == b["shape"][1] and _profiles_complement(a["profile"], b["profile"]):
                    specs.append({"members": (i, j), "kind": "H", "top": j, "bottom": i})
    return specs


def _pick_matching(pieces: list[dict], specs: list[dict]) -> list[dict]:
    by_piece: dict[int, list[int]] = {i: [] for i in range(len(pieces))}
    for idx, spec in enumerate(specs):
        i, j = spec["members"]
        by_piece[int(i)].append(int(idx))
        by_piece[int(j)].append(int(idx))

    def search(remaining: frozenset[int]) -> list[int] | None:
        if not remaining:
            return []
        pivot = min(
            remaining,
            key=lambda i: len(
                [
                    spec_idx
                    for spec_idx in by_piece[int(i)]
                    if all(member in remaining for member in specs[spec_idx]["members"])
                ]
            ),
        )
        choices = [
            spec_idx
            for spec_idx in by_piece[int(pivot)]
            if all(member in remaining for member in specs[spec_idx]["members"])
        ]
        if not choices:
            return None
        for spec_idx in choices:
            i, j = specs[spec_idx]["members"]
            tail = search(remaining - {int(i), int(j)})
            if tail is not None:
                return [int(spec_idx)] + tail
        return None

    chosen = search(frozenset(range(len(pieces))))
    assert chosen is not None, "expected a complete complementary matching"
    return [specs[idx] for idx in chosen]


def _paste_non_overlapping(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == patch.shape, "patch would go out of bounds"
    overlap = (target != BLACK) & (patch != BLACK)
    assert not bool(overlap.any()), "assembled pieces must not overlap"
    target[patch != BLACK] = patch[patch != BLACK]


def _assemble_pair(pieces: list[dict], spec: dict) -> dict:
    i, j = spec["members"]
    r0 = min(int(pieces[i]["bbox"][0]), int(pieces[j]["bbox"][0]))
    c0 = min(int(pieces[i]["bbox"][1]), int(pieces[j]["bbox"][1]))

    if spec["kind"] == "V":
        left = pieces[int(spec["left"])]
        right = pieces[int(spec["right"])]
        k = int(left["profile"][0] + right["profile"][0])
        offset = int(left["shape"][1] - k)
        out_h = int(left["shape"][0])
        out_w = int(offset + right["shape"][1])
        rect = np.full((out_h, out_w), BLACK, dtype=int)
        _paste_non_overlapping(rect, left["patch"], 0, 0)
        _paste_non_overlapping(rect, right["patch"], 0, offset)
        assert not bool((rect == BLACK).any()), "assembled rectangle must be solid"
        rect = np.rot90(rect, 1)
    else:
        top = pieces[int(spec["top"])]
        bottom = pieces[int(spec["bottom"])]
        k = int(top["profile"][0] + bottom["profile"][0])
        offset = int(top["shape"][0] - k)
        out_h = int(offset + bottom["shape"][0])
        out_w = int(top["shape"][1])
        rect = np.full((out_h, out_w), BLACK, dtype=int)
        _paste_non_overlapping(rect, top["patch"], 0, 0)
        _paste_non_overlapping(rect, bottom["patch"], offset, 0)
        assert not bool((rect == BLACK).any()), "assembled rectangle must be solid"

    area = int(rect.shape[0] * rect.shape[1])
    return {"patch": rect, "area": area, "anchor": (r0, c0)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered toothed fragments into solid rectangles, rotate each merged
    rectangle into its canonical upright orientation, sort by area, and stack
    them in the output zone to the right of the grey separator.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep = _find_separator(g)

    left_mask = (g[:, :sep] != BLACK) & (g[:, :sep] != GREY)
    comps = _components4(left_mask)
    assert comps and len(comps) % 2 == 0, "expected an even number of fragments"

    pieces: list[dict] = []
    for coords in comps:
        full_coords = np.asarray([(int(r), int(c)) for r, c in coords], dtype=int)
        r0, c0, r1, c1 = _bbox(full_coords)
        patch = np.asarray(g[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        pieces.append(_piece_info(patch, full_coords))

    specs = _candidate_specs(pieces)
    paired = _pick_matching(pieces, specs)
    rectangles = [_assemble_pair(pieces, spec) for spec in paired]
    rectangles.sort(key=lambda item: (int(item["area"]), int(item["anchor"][0]), int(item["anchor"][1])))

    out = np.full((h, w), BLACK, dtype=int)
    out[:, sep] = GREY
    start_c = int(sep + 2)
    row = 0
    for item in rectangles:
        patch = item["patch"]
        ph, pw = patch.shape
        assert row + ph <= h, "stacked output exceeds canvas height"
        assert start_c + pw <= w, "stacked output exceeds canvas width"
        out[row : row + ph, start_c : start_c + pw] = patch
        row += int(ph + 1)
    return out


def _canonical_profile(bits: tuple[int, ...]) -> tuple[int, ...]:
    comp = tuple(1 - int(v) for v in bits)
    return min(tuple(int(v) for v in bits), comp)


def _random_teeth(length: int) -> tuple[int, ...]:
    count = random.randint(1, min(3, length - 1))
    return tuple(sorted(random.sample(range(length), count)))


def _split_vertical(h: int, w: int, color_a: int, color_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    teeth = _random_teeth(h)
    indicator = np.zeros(h, dtype=int)
    indicator[list(teeth)] = 1
    left_tabs = random.random() < 0.5
    cut_lo = 3
    cut_hi = w - 3
    assert cut_lo <= cut_hi
    cut = random.randint(cut_lo, cut_hi)
    boundary = cut + indicator if left_tabs else cut - indicator
    assert int(boundary.min()) >= 2 and int(boundary.max()) <= w - 2

    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    for r, stop in enumerate(boundary):
        left[int(r), : int(stop)] = True
        right[int(r), int(stop) :] = True

    left_crop = _crop_mask(left)
    right_crop = _crop_mask(right)
    rect = np.full((h, w), BLACK, dtype=int)
    rect[left] = int(color_a)
    rect[right] = int(color_b)
    return _paint_mask(left_crop, color_a), _paint_mask(right_crop, color_b), np.rot90(rect, 1), tuple(int(v) for v in indicator)


def _split_horizontal(h: int, w: int, color_a: int, color_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    teeth = _random_teeth(w)
    indicator = np.zeros(w, dtype=int)
    indicator[list(teeth)] = 1
    top_tabs = random.random() < 0.5
    cut_lo = 3
    cut_hi = h - 3
    assert cut_lo <= cut_hi
    cut = random.randint(cut_lo, cut_hi)
    boundary = cut + indicator if top_tabs else cut - indicator
    assert int(boundary.min()) >= 2 and int(boundary.max()) <= h - 2

    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    for c, stop in enumerate(boundary):
        top[: int(stop), int(c)] = True
        bottom[int(stop) :, int(c)] = True

    top_crop = _crop_mask(top)
    bottom_crop = _crop_mask(bottom)
    rect = np.full((h, w), BLACK, dtype=int)
    rect[top] = int(color_a)
    rect[bottom] = int(color_b)
    return _paint_mask(top_crop, color_a), _paint_mask(bottom_crop, color_b), rect, tuple(int(v) for v in indicator)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_scattered(patches: list[np.ndarray], h: int, left_w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        row_lo, row_hi = 1, h - ph - 1
        col_lo, col_hi = 1, left_w - pw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        placed = False
        for _ in range(160):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_overlap(box, other, gap=1) for other in boxes):
                continue
            boxes.append(box)
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [placements[i] for i in range(len(patches))]


def _build_expected(shape: tuple[int, int], sep: int, rects: list[np.ndarray]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    out[:, sep] = GREY
    row = 0
    for rect in sorted(rects, key=lambda patch: int(patch.shape[0] * patch.shape[1])):
        h, w = rect.shape
        out[row : row + h, sep + 2 : sep + 2 + w] = rect
        row += int(h + 1)
    return out


def _generate_attempt() -> np.ndarray | None:
    n_pairs = random.choice((2, 3))
    colors = random.sample(PALETTE, 2 * n_pairs)
    pair_patches: list[np.ndarray] = []
    output_rects: list[np.ndarray] = []
    used_areas: set[int] = set()
    used_signatures: set[tuple[str, int, tuple[int, ...]]] = set()

    for pair_idx in range(n_pairs):
        color_a = int(colors[2 * pair_idx])
        color_b = int(colors[2 * pair_idx + 1])
        built = False
        for _ in range(80):
            kind = random.choice(("V", "H"))
            if kind == "V":
                h = random.randint(4, 8)
                w = random.randint(6, 8)
            else:
                h = random.randint(6, 8)
                w = random.randint(4, 8)
            area = int(h * w)
            if area in used_areas:
                continue

            if kind == "V":
                piece_a, piece_b, rect, bits = _split_vertical(h, w, color_a, color_b)
                signature = ("V", h, _canonical_profile(bits))
            else:
                piece_a, piece_b, rect, bits = _split_horizontal(h, w, color_a, color_b)
                signature = ("H", w, _canonical_profile(bits))

            if signature in used_signatures:
                continue

            used_areas.add(area)
            used_signatures.add(signature)
            pair_patches.extend([piece_a, piece_b])
            output_rects.append(rect)
            built = True
            break
        if not built:
            return None

    total_out_h = sum(int(rect.shape[0]) for rect in output_rects) + len(output_rects) - 1
    max_out_w = max(int(rect.shape[1]) for rect in output_rects)
    max_piece_h = max(int(patch.shape[0]) for patch in pair_patches)
    max_piece_w = max(int(patch.shape[1]) for patch in pair_patches)

    min_h = max(total_out_h, max_piece_h + 2, 16)
    max_h = min(MAX_GRID, min_h + 6)
    if min_h > max_h:
        return None
    grid_h = random.randint(min_h, max_h)

    left_w_min = max(max_piece_w + 3, 11)
    left_w_max = MAX_GRID - (2 + max_out_w + 1)
    if left_w_min > left_w_max:
        return None
    left_w = random.randint(left_w_min, left_w_max)
    grid_w = int(left_w + 2 + max_out_w + random.randint(1, 3))
    if grid_w > MAX_GRID:
        return None

    placements = _place_scattered(pair_patches, grid_h, left_w)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    sep = int(left_w)
    grid[:, sep] = GREY
    for patch, (r0, c0) in zip(pair_patches, placements):
        ph, pw = patch.shape
        target = grid[r0 : r0 + ph, c0 : c0 + pw]
        if target.shape != patch.shape or bool(((target != BLACK) & (patch != BLACK)).any()):
            return None
        target[patch != BLACK] = patch[patch != BLACK]

    expected = _build_expected(grid.shape, sep, output_rects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _generate_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid toothed-fragment puzzle")
