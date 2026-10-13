from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
ARC_MAX = 30
COLOR_POOL = tuple(range(1, 10))
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
DIM_CANDIDATES = [
    (h, w)
    for h in range(2, 6)
    for w in range(h, 9)
    if 6 <= h * w <= 20
]


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert rows.any() and cols.any()
    return np.asarray(mask[np.ix_(rows, cols)], dtype=bool)


def _paint_from_labels(labels: np.ndarray, color1: int, color2: int) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=int)
    out[labels == 1] = int(color1)
    out[labels == 2] = int(color2)
    return out


def _canonicalize_label_patch(labels: np.ndarray) -> np.ndarray:
    best_key = None
    best = None
    for k in range(4):
        rot = np.rot90(labels, k)
        if rot.shape[0] > rot.shape[1]:
            continue
        arr = np.asarray(rot, dtype=np.uint8)
        key = (int(arr.shape[0]), int(arr.shape[1]), arr.tobytes())
        if best_key is None or key < best_key:
            best_key = key
            best = arr.copy()
    assert best is not None
    return best


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == BG or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
            for r, c in coords:
                mask[r - min_r, c - min_c] = True

            components.append(
                {
                    "color": color,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))  # type: ignore[index]
    return components


def _iter_label_merges(
    fixed_mask: np.ndarray,
    fixed_label: int,
    moving_mask: np.ndarray,
    moving_label: int,
) -> list[np.ndarray]:
    fixed_cells = [tuple(map(int, rc)) for rc in np.argwhere(fixed_mask)]
    fixed_set = set(fixed_cells)
    fh, fw = map(int, fixed_mask.shape)
    total_area = len(fixed_cells) + int(moving_mask.sum())
    patches: list[np.ndarray] = []

    for rot in range(4):
        rotated = np.rot90(moving_mask, rot)
        mh, mw = map(int, rotated.shape)
        moving_cells = [tuple(map(int, rc)) for rc in np.argwhere(rotated)]

        for dr in range(-mh, fh + 1):
            for dc in range(-mw, fw + 1):
                overlap = False
                for r, c in moving_cells:
                    if (r + dr, c + dc) in fixed_set:
                        overlap = True
                        break
                if overlap:
                    continue

                min_r = min(0, dr)
                min_c = min(0, dc)
                max_r = max(fh - 1, dr + mh - 1)
                max_c = max(fw - 1, dc + mw - 1)
                out_h = max_r - min_r + 1
                out_w = max_c - min_c + 1
                if out_h * out_w != total_area:
                    continue

                patch = np.zeros((out_h, out_w), dtype=np.uint8)
                fixed_r0 = -min_r
                fixed_c0 = -min_c
                for r, c in fixed_cells:
                    patch[fixed_r0 + r, fixed_c0 + c] = fixed_label

                moving_r0 = dr - min_r
                moving_c0 = dc - min_c
                for r, c in moving_cells:
                    patch[moving_r0 + r, moving_c0 + c] = moving_label
                patches.append(patch)

    return patches


def _best_rectangle_merge(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray | None:
    best_key = None
    best_patch = None

    for patch in _iter_label_merges(mask_a, 1, mask_b, 2):
        canon = _canonicalize_label_patch(patch)
        key = (int(canon.shape[0]), int(canon.shape[1]), canon.tobytes())
        if best_key is None or key < best_key:
            best_key = key
            best_patch = canon.copy()

    for patch in _iter_label_merges(mask_b, 2, mask_a, 1):
        canon = _canonicalize_label_patch(patch)
        key = (int(canon.shape[0]), int(canon.shape[1]), canon.tobytes())
        if best_key is None or key < best_key:
            best_key = key
            best_patch = canon.copy()

    return best_patch


def _solve_matching(components: list[dict[str, object]]) -> list[tuple[int, int]]:
    n = len(components)
    assert n > 0 and n % 2 == 0

    edges: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            patch = _best_rectangle_merge(
                np.asarray(components[i]["mask"], dtype=bool),  # type: ignore[index]
                np.asarray(components[j]["mask"], dtype=bool),  # type: ignore[index]
            )
            if patch is not None:
                edges[(i, j)] = patch

    @lru_cache(maxsize=None)
    def rec(mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if mask == 0:
            return 1, ()

        first = 0
        while ((mask >> first) & 1) == 0:
            first += 1

        count = 0
        sample: tuple[tuple[int, int], ...] = ()
        base = mask ^ (1 << first)
        for j in range(first + 1, n):
            if ((base >> j) & 1) == 0:
                continue
            if (first, j) not in edges:
                continue

            sub_count, sub_solution = rec(base ^ (1 << j))
            if sub_count == 0:
                continue
            if count == 0:
                sample = ((first, j),) + sub_solution
            count = min(2, count + sub_count)

        return count, sample

    total_count, solution = rec((1 << n) - 1)
    assert total_count == 1, "pairing relation must yield one perfect matching"
    return list(solution)


def _pair_patches(components: list[dict[str, object]], matching: list[tuple[int, int]]) -> list[np.ndarray]:
    merged = []
    for i, j in matching:
        labels = _best_rectangle_merge(
            np.asarray(components[i]["mask"], dtype=bool),  # type: ignore[index]
            np.asarray(components[j]["mask"], dtype=bool),  # type: ignore[index]
        )
        assert labels is not None
        patch = _paint_from_labels(
            labels,
            int(components[i]["color"]),  # type: ignore[index]
            int(components[j]["color"]),  # type: ignore[index]
        )
        merged.append(patch)
    return merged


def _pack_output(patches: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(
        patches,
        key=lambda patch: (int(patch.shape[0] * patch.shape[1]), int(patch.shape[1])),
    )
    out_h = max(int(patch.shape[0]) for patch in ordered)
    out_w = sum(int(patch.shape[1]) for patch in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c = 0
    for patch in ordered:
        h, w = map(int, patch.shape)
        out[:h, c : c + w] = patch
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, reassemble their two-color rectangles, and sort the rectangles by size.

    Input structure:
    - The background is black.
    - Several non-touching monochrome 4-connected objects are scattered across the grid.
    - The objects come in unique pairs: after rotating one piece by 0/90/180/270 degrees, the two shapes can tile a solid rectangle.

    Rule:
    - Extract every object and test every pair by rotating and translating the shapes until their union is a filled rectangle with no overlap.
    - Recover the unique perfect pairing, build each merged rectangle, and keep each cell in the original color of its source fragment.
    - Normalize each merged rectangle to a width-at-least-height orientation, then sort the rectangles by area and, for equal areas, by width.
    - Place the sorted rectangles left-to-right, top-aligned, with one black separator column between neighbors.

    Key insight:
    - Colors do not identify partners; the signal is geometric complementarity: two disconnected shapes are matched only if they exactly complete one rectangle together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= ARC_MAX

    components = _find_components(g)
    assert components, "input must contain foreground objects"

    matching = _solve_matching(components)
    merged = _pair_patches(components, matching)
    out = _pack_output(merged)
    assert max(out.shape) <= ARC_MAX
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _make_row_split_labels(h: int, w: int) -> np.ndarray:
    labels = np.zeros((h, w), dtype=np.uint8)
    cuts = [random.randint(1, w - 1) for _ in range(h)]
    if w >= 4 and random.random() < 0.7 and len(set(cuts)) == 1:
        cuts[random.randrange(h)] = random.choice([c for c in range(1, w) if c != cuts[0]])
    for r, cut in enumerate(cuts):
        labels[r, :cut] = 1
        labels[r, cut:] = 2
    return labels


def _make_col_split_labels(h: int, w: int) -> np.ndarray:
    labels = np.zeros((h, w), dtype=np.uint8)
    cuts = [random.randint(1, h - 1) for _ in range(w)]
    if h >= 4 and random.random() < 0.7 and len(set(cuts)) == 1:
        cuts[random.randrange(w)] = random.choice([c for c in range(1, h) if c != cuts[0]])
    for c, cut in enumerate(cuts):
        labels[:cut, c] = 1
        labels[cut:, c] = 2
    return labels


def _make_pair_definition(h: int, w: int, color1: int, color2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        labels = _make_row_split_labels(h, w)
    else:
        labels = _make_col_split_labels(h, w)

    mask1 = _tight_crop(labels == 1)
    mask2 = _tight_crop(labels == 2)
    assert mask1.any() and mask2.any()

    canon_labels = _canonicalize_label_patch(labels)
    expected = _paint_from_labels(canon_labels, color1, color2)
    return mask1, mask2, expected


def _rotated_piece(mask: np.ndarray, color: int) -> np.ndarray:
    rot = random.randint(0, 3)
    rotated = np.rot90(mask, rot)
    piece = np.zeros(rotated.shape, dtype=int)
    piece[rotated] = int(color)
    return piece


def _sample_dims(pair_count: int, force_tie: bool) -> list[tuple[int, int]] | None:
    by_area: dict[int, list[tuple[int, int]]] = {}
    for dims in DIM_CANDIDATES:
        by_area.setdefault(dims[0] * dims[1], []).append(dims)

    for _ in range(200):
        chosen: list[tuple[int, int]] = []
        used_areas: set[int] = set()

        if force_tie:
            tie_areas = [area for area, dims_list in by_area.items() if len(dims_list) >= 2]
            random.shuffle(tie_areas)
            if not tie_areas:
                return None
            tie_area = tie_areas[0]
            first, second = random.sample(by_area[tie_area], 2)
            chosen.extend([first, second])
            used_areas.add(tie_area)

        remaining = DIM_CANDIDATES[:]
        random.shuffle(remaining)
        for dims in remaining:
            area = dims[0] * dims[1]
            if len(chosen) >= pair_count:
                break
            if dims in chosen:
                continue
            if area in used_areas:
                continue
            chosen.append(dims)
            used_areas.add(area)

        if len(chosen) != pair_count:
            continue

        widths = [w for _, w in chosen]
        if sum(widths) + pair_count - 1 > ARC_MAX:
            continue
        return chosen

    return None


def _place_pieces(pieces: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(piece.shape[0]) for piece in pieces)
    max_w = max(int(piece.shape[1]) for piece in pieces)
    total_bbox_area = sum(int(piece.shape[0] * piece.shape[1]) for piece in pieces)

    min_h = max(max_h + 2, 14, int(np.ceil(np.sqrt(total_bbox_area * 1.4))))
    min_w = max(max_w + 2, 14, int(np.ceil(np.sqrt(total_bbox_area * 1.4))))
    if min_h > ARC_MAX or min_w > ARC_MAX:
        return None

    for _ in range(80):
        h = random.randint(min_h, ARC_MAX)
        w = random.randint(min_w, ARC_MAX)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        order = list(range(len(pieces)))
        random.shuffle(order)
        order.sort(key=lambda idx: int(pieces[idx].shape[0] * pieces[idx].shape[1]), reverse=True)

        ok = True
        for idx in order:
            piece = pieces[idx]
            ph, pw = map(int, piece.shape)
            piece_mask = piece != BG
            blocked = _dilate8(occupied)

            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                r0 = random.randint(0, h - ph)
                c0 = random.randint(0, w - pw)
                if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & piece_mask).any()):
                    continue
                sub = grid[r0 : r0 + ph, c0 : c0 + pw]
                sub[piece_mask] = piece[piece_mask]
                occupied[r0 : r0 + ph, c0 : c0 + pw] |= piece_mask
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok:
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair_count = random.randint(2, 4)
        force_tie = random.random() < 0.5
        dims = _sample_dims(pair_count, force_tie)
        if dims is None:
            continue

        colors = random.sample(COLOR_POOL, 2 * pair_count)
        expected_patches = []
        pieces = []

        for i, (h, w) in enumerate(dims):
            color1 = colors[2 * i]
            color2 = colors[2 * i + 1]
            mask1, mask2, expected = _make_pair_definition(h, w, color1, color2)
            expected_patches.append(expected)
            pieces.append(_rotated_piece(mask1, color1))
            pieces.append(_rotated_piece(mask2, color2))

        expected = _pack_output(expected_patches)
        if max(expected.shape) > ARC_MAX:
            continue

        grid = _place_pieces(pieces)
        if grid is None:
            continue

        try:
            if not np.array_equal(main(grid), expected):
                continue
        except AssertionError:
            continue

        return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
