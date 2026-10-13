from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

# Generator configuration.
PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 7
INPUT_SIDE_MIN = 16
INPUT_SIDE_MAX = 30
OUTPUT_WIDTH_MAX = 30
PLACEMENT_PADDING = 1
MAX_PAIR_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), np.asarray(mask, dtype=bool).tobytes()


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    r0 = int(rows[0])
    r1 = int(rows[-1])
    c0 = int(cols[0])
    c1 = int(cols[-1])
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool), (r0, c0)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or bool(seen[r, c]):
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                    "key": _mask_key(mask),
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


def _assemble_pairs(pairs: list[tuple[int, int, np.ndarray, tuple[int, int]]], components: list[dict]) -> list[np.ndarray]:
    rectangles: list[tuple[int, int, int, np.ndarray]] = []
    for outer_idx, filler_idx, hole_mask, offset in pairs:
        outer = components[outer_idx]
        filler = components[filler_idx]
        outer_mask = np.asarray(outer["mask"], dtype=bool)
        filler_mask = np.asarray(filler["mask"], dtype=bool)
        assert _mask_key(hole_mask) == filler["key"]

        rect = np.zeros(outer_mask.shape, dtype=int)
        rect[outer_mask] = int(outer["color"])

        off_r, off_c = offset
        hh, ww = filler_mask.shape
        view = rect[off_r : off_r + hh, off_c : off_c + ww]
        view[filler_mask] = int(filler["color"])
        assert not bool((rect == BLACK).any())

        area = int(rect.shape[0] * rect.shape[1])
        r0, c0, _r1, _c1 = outer["bbox"]
        rectangles.append((area, r0, c0, rect))

    rectangles.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [rect for _area, _r0, _c0, rect in rectangles]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, reconstruct their rectangles, and pack the rectangles from largest to smallest.

    Input structure:
    - The background is black.
    - The foreground consists of an even number of scattered monochrome 4-connected fragments.
    - Each fragment pair has different colors, and one fragment's negative space inside its tight box matches the other's tight shape exactly.

    Rule:
    - Find every monochrome 4-connected fragment and extract its tight boolean shape.
    - For each fragment A, take the unoccupied cells inside A's tight bounding box, crop that negative-space mask tightly, and match it with the tight shape of another fragment B.
    - Reconstruct each pair as the full rectangle defined by A's box: keep A's pixels in A's color and fill the cropped negative-space pixels with B's color.
    - Sort the reconstructed rectangles by area from largest to smallest, then place them left to right with one black separator column and with their top edges aligned.

    Key insight:
    - The pairing is determined by exact shape complementarity of negative space, not by color or proximity, and the output order is determined by the completed rectangle areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components and len(components) % 2 == 0

    key_to_indices: defaultdict[tuple[tuple[int, int], bytes], list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        key_to_indices[comp["key"]].append(idx)

    pairs: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    usage = Counter()
    for outer_idx, outer in enumerate(components):
        hole = _crop_mask(~np.asarray(outer["mask"], dtype=bool))
        if hole is None:
            continue
        hole_mask, offset = hole
        matches = [
            filler_idx
            for filler_idx in key_to_indices[_mask_key(hole_mask)]
            if filler_idx != outer_idx and int(components[filler_idx]["color"]) != int(outer["color"])
        ]
        if not matches:
            continue
        assert len(matches) == 1
        filler_idx = int(matches[0])
        pairs.append((outer_idx, filler_idx, hole_mask, offset))
        usage[outer_idx] += 1
        usage[filler_idx] += 1

    assert len(pairs) * 2 == len(components)
    assert len({outer_idx for outer_idx, _filler_idx, _hole, _offset in pairs}) == len(pairs)
    assert len({filler_idx for _outer_idx, filler_idx, _hole, _offset in pairs}) == len(pairs)
    assert len(usage) == len(components)
    assert all(count == 1 for count in usage.values())

    rectangles = _assemble_pairs(pairs, components)
    out_h = max(rect.shape[0] for rect in rectangles)
    out_w = sum(rect.shape[1] for rect in rectangles) + len(rectangles) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for rect in rectangles:
        rh, rw = rect.shape
        out[:rh, col : col + rw] = rect
        col += rw + 1

    return out


def _random_spanning_mask(h: int, w: int, area: int) -> np.ndarray:
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)

    mask = np.zeros((h, w), dtype=bool)
    pivot_r = int(RNG.integers(0, h))
    pivot_c = int(RNG.integers(0, w))
    mask[pivot_r, :] = True
    mask[:, pivot_c] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        choices = list(frontier)
        nr, nc = choices[int(RNG.integers(0, len(choices)))]
        mask[nr, nc] = True

    return mask


def _random_hole_mask(h: int, w: int) -> np.ndarray:
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)

    min_area = h + w - 1
    max_area = h * w
    for _ in range(MAX_PAIR_ATTEMPTS):
        area = int(RNG.integers(min_area, max_area + 1))
        mask = _random_spanning_mask(h, w, area)
        if _is_connected(mask) and _crop_mask(mask) is not None and _crop_mask(mask)[0].shape == mask.shape:
            return mask
    raise RuntimeError("failed to build a spanning hole mask")


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _sample_pair(outer_color: int, filler_color: int) -> dict | None:
    for _ in range(MAX_PAIR_ATTEMPTS):
        rect_h = int(RNG.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        rect_w = int(RNG.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        hole_h = int(RNG.integers(1, rect_h - 1))
        hole_w = int(RNG.integers(1, rect_w - 1))
        if hole_h > rect_h - 2 or hole_w > rect_w - 2:
            continue

        hole_mask = _random_hole_mask(hole_h, hole_w)
        off_r = int(RNG.integers(1, rect_h - hole_h))
        off_c = int(RNG.integers(1, rect_w - hole_w))
        if off_r + hole_h >= rect_h or off_c + hole_w >= rect_w:
            continue

        outer_mask = np.ones((rect_h, rect_w), dtype=bool)
        outer_mask[off_r : off_r + hole_h, off_c : off_c + hole_w][hole_mask] = False
        if not _is_connected(outer_mask):
            continue

        rect = np.zeros((rect_h, rect_w), dtype=int)
        rect[outer_mask] = int(outer_color)
        rect[off_r : off_r + hole_h, off_c : off_c + hole_w][hole_mask] = int(filler_color)
        if bool((rect == BLACK).any()):
            continue

        return {
            "area": rect_h * rect_w,
            "outer_patch": _paint_mask(outer_mask, outer_color),
            "filler_patch": _paint_mask(hole_mask, filler_color),
            "assembled": rect,
        }

    return None


def _render_expected(pairs: list[dict]) -> np.ndarray:
    ordered = sorted(enumerate(pairs), key=lambda item: (-item[1]["area"], item[0]))
    rects = [pair["assembled"] for _idx, pair in ordered]
    out_h = max(rect.shape[0] for rect in rects)
    out_w = sum(rect.shape[1] for rect in rects) + len(rects) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for rect in rects:
        rh, rw = rect.shape
        out[:rh, col : col + rw] = rect
        col += rw + 1
    return out


def _place_patches(patches: list[np.ndarray], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = list(range(len(patches)))
    RNG.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                rs = max(0, r0 - PLACEMENT_PADDING)
                re = min(h, r0 + ph + PLACEMENT_PADDING)
                cs = max(0, c0 - PLACEMENT_PADDING)
                ce = min(w, c0 + pw + PLACEMENT_PADDING)
                if bool(occupied[rs:re, cs:ce].any()):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = candidates[int(RNG.integers(0, len(candidates)))]
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BLACK
        view[mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True

    return grid


def _sample_canvas(patches: list[np.ndarray]) -> tuple[int, int]:
    padded_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    min_side = int(np.ceil(np.sqrt(padded_area)))
    max_h = max(patch.shape[0] for patch in patches) + 2
    max_w = max(patch.shape[1] for patch in patches) + 2
    low_h = max(INPUT_SIDE_MIN, max_h, min_side - 2)
    low_w = max(INPUT_SIDE_MIN, max_w, min_side - 2)
    h = int(RNG.integers(low_h, INPUT_SIDE_MAX + 1))
    w = int(RNG.integers(low_w, INPUT_SIDE_MAX + 1))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice(PAIR_COUNT_CHOICES))
    colors = RNG.choice(np.arange(1, 10), size=2 * pair_count, replace=False)

    pairs: list[dict] = []
    used_areas: set[int] = set()
    for i in range(pair_count):
        pair = None
        for _ in range(MAX_PAIR_ATTEMPTS):
            pair = _sample_pair(int(colors[2 * i]), int(colors[2 * i + 1]))
            if pair is None:
                continue
            if int(pair["area"]) in used_areas:
                pair = None
                continue
            pairs.append(pair)
            used_areas.add(int(pairs[-1]["area"]))
            break
        if pair is None:
            return None

    expected = _render_expected(pairs)
    if expected.shape[1] > OUTPUT_WIDTH_MAX:
        return None

    patches: list[np.ndarray] = []
    for pair in pairs:
        patches.append(np.asarray(pair["outer_patch"], dtype=int))
        patches.append(np.asarray(pair["filler_patch"], dtype=int))

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        canvas_h, canvas_w = _sample_canvas(patches)
        grid = _place_patches(patches, canvas_h, canvas_w)
        if grid is None:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
