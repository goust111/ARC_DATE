from __future__ import annotations

import random

import numpy as np


BG = 0
ANCHOR = 2
FRAGMENT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
PAIR_DIM_POOL = (
    (4, 4),
    (5, 4),
    (6, 4),
    (5, 5),
    (7, 4),
    (6, 5),
)
MAX_ATTEMPTS = 400


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _find_monochrome_components(grid: np.ndarray, ignore: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


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


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments by bounding-box shape and redraw each completed rectangle at a red anchor.

    Input structure:
    - Black background with scattered single-color fragments and red single-pixel anchors.
    - Fragments come in pairs whose minimal bounding boxes have the same size and whose filled cells are exact complements inside that box.
    - Each completed rectangle area is unique, so the pairs can be ordered unambiguously by size.

    Rule:
    - Extract every non-black, non-red fragment, normalize it to its own minimal bounding box, and pair fragments with equal box sizes whose masks are exact complements.
    - Sort the paired rectangles by area from small to large, and sort the red anchors in row-major order.
    - Clear the original fragments and anchors, then redraw each completed rectangle with its assigned anchor as the rectangle's top-left corner, preserving the source colors on their own mask cells.

    Key insight:
    Pairing is determined by complementary occupancy inside matching bounding boxes, while placement is determined independently by rectangle-area order versus anchor order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    anchors = sorted((int(r), int(c)) for r, c in np.argwhere(g == ANCHOR))
    comps = _find_monochrome_components(g, ignore={BG, ANCHOR})
    assert len(comps) % 2 == 0 and len(comps) > 0
    assert len(anchors) * 2 == len(comps)

    frags: list[dict[str, object]] = []
    for comp in comps:
        coords = comp["coords"]
        assert isinstance(coords, list)
        box = _bbox(coords)
        mask = _mask_from_coords(coords, box)
        frags.append(
            {
                "color": int(comp["color"]),
                "bbox": box,
                "shape": tuple(map(int, mask.shape)),
                "mask": mask,
            }
        )

    used: set[int] = set()
    pairs: list[dict[str, object]] = []
    for i, frag_a in enumerate(frags):
        if i in used:
            continue
        matches: list[int] = []
        for j, frag_b in enumerate(frags):
            if i == j or j in used:
                continue
            if frag_a["shape"] != frag_b["shape"]:
                continue
            if int(frag_a["color"]) == int(frag_b["color"]):
                continue
            if np.array_equal(frag_a["mask"], ~frag_b["mask"]):
                matches.append(j)
        assert len(matches) == 1
        j = matches[0]
        used.add(i)
        used.add(j)
        pair_h, pair_w = frag_a["shape"]
        pairs.append(
            {
                "area": int(pair_h * pair_w),
                "h": int(pair_h),
                "w": int(pair_w),
                "mask_a": frag_a["mask"],
                "color_a": int(frag_a["color"]),
                "color_b": int(frags[j]["color"]),
            }
        )
    assert len(used) == len(frags)

    pairs.sort(key=lambda item: int(item["area"]))
    assert len(pairs) == len(anchors)

    out = np.full((h, w), BG, dtype=int)
    for pair, (r0, c0) in zip(pairs, anchors):
        ph = int(pair["h"])
        pw = int(pair["w"])
        assert r0 + ph <= h and c0 + pw <= w
        patch = np.where(pair["mask_a"], int(pair["color_a"]), int(pair["color_b"])).astype(int)
        region = out[r0 : r0 + ph, c0 : c0 + pw]
        assert np.all(region == BG)
        region[:, :] = patch
    return out


def _random_path_mask(base_h: int, base_w: int) -> np.ndarray:
    assert base_h >= base_w >= 2
    steps = [1] * (base_w - 1) + [0] * (base_h - base_w)
    random.shuffle(steps)
    cols = [0]
    for step in steps:
        cols.append(cols[-1] + step)
    assert len(cols) == base_h and cols[-1] == base_w - 1

    mask = np.zeros((base_h, base_w), dtype=bool)
    for r, c in enumerate(cols):
        mask[r, c] = True
    return mask


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if out.shape[0] != out.shape[1] and random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _all_candidates(blocked: np.ndarray, footprint: np.ndarray) -> list[tuple[int, int]]:
    gh, gw = blocked.shape
    ph, pw = footprint.shape
    candidates: list[tuple[int, int]] = []
    for r in range(gh - ph + 1):
        for c in range(gw - pw + 1):
            if not bool((blocked[r : r + ph, c : c + pw] & footprint).any()):
                candidates.append((r, c))
    return candidates


def _mark(blocked: np.ndarray, footprint: np.ndarray, top: int, left: int) -> None:
    ph, pw = footprint.shape
    blocked[top : top + ph, left : left + pw] |= footprint


def _rectangle_footprint(h: int, w: int) -> np.ndarray:
    return _dilate8(np.ones((h, w), dtype=bool))


def _build_pair() -> dict[str, object]:
    base_h, base_w = random.choice(PAIR_DIM_POOL)
    mask_a = _transform_mask(_random_path_mask(base_h, base_w))
    mask_b = ~mask_a
    if random.random() < 0.5:
        mask_a, mask_b = mask_b, mask_a
    h, w = mask_a.shape
    return {
        "h": int(h),
        "w": int(w),
        "area": int(h * w),
        "mask_a": mask_a,
        "mask_b": mask_b,
    }


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_attempt() -> np.ndarray | None:
    num_pairs = random.choice((2, 3))
    chosen_dims = random.sample(PAIR_DIM_POOL, num_pairs)
    pairs = []
    for base_h, base_w in chosen_dims:
        mask_a = _transform_mask(_random_path_mask(base_h, base_w))
        mask_b = ~mask_a
        if random.random() < 0.5:
            mask_a, mask_b = mask_b, mask_a
        h, w = mask_a.shape
        pairs.append(
            {
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
                "mask_a": mask_a,
                "mask_b": mask_b,
            }
        )

    colors = random.sample(FRAGMENT_COLORS, 2 * num_pairs)
    random.shuffle(colors)
    for i, pair in enumerate(pairs):
        pair["color_a"] = int(colors[2 * i])
        pair["color_b"] = int(colors[2 * i + 1])

    max_h = max(int(pair["h"]) for pair in pairs)
    max_w = max(int(pair["w"]) for pair in pairs)

    min_side = max(20, max_h + 6, max_w + 6)
    if min_side > 30:
        return None
    grid_h = random.randint(min_side, 30)
    grid_w = random.randint(min_side, 30)

    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    anchor_positions: list[tuple[int, int]] = []
    anchor_footprint = _rectangle_footprint(max_h, max_w)
    anchor_candidates = _all_candidates(reserved, anchor_footprint)
    if len(anchor_candidates) < num_pairs:
        return None
    for _ in range(num_pairs):
        candidates = _all_candidates(reserved, anchor_footprint)
        if not candidates:
            return None
        pos = random.choice(candidates)
        anchor_positions.append(pos)
        _mark(reserved, anchor_footprint, pos[0], pos[1])

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    for r, c in anchor_positions:
        inp[r, c] = ANCHOR

    blocked = reserved.copy()
    placements: list[tuple[np.ndarray, int, int]] = []
    fragment_specs: list[tuple[np.ndarray, int]] = []
    for pair in pairs:
        fragment_specs.append((_patch_from_mask(pair["mask_a"], int(pair["color_a"])), int(np.sum(pair["mask_a"]))))
        fragment_specs.append((_patch_from_mask(pair["mask_b"], int(pair["color_b"])), int(np.sum(pair["mask_b"]))))
    fragment_specs.sort(key=lambda item: (item[0].shape[0] * item[0].shape[1], item[1]), reverse=True)

    for patch, _size in fragment_specs:
        footprint = _dilate8(patch != BG)
        candidates = _all_candidates(blocked, footprint)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements.append((patch, top, left))
        _mark(blocked, footprint, top, left)
        _paint_patch(inp, patch, top, left)

    out = np.full_like(inp, BG)
    sorted_pairs = sorted(pairs, key=lambda item: int(item["area"]))
    sorted_anchors = sorted(anchor_positions)
    for pair, (r0, c0) in zip(sorted_pairs, sorted_anchors):
        patch = np.where(pair["mask_a"], int(pair["color_a"]), int(pair["color_b"])).astype(int)
        _paint_patch(out, patch, r0, c0)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
