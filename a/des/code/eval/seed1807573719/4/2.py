from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components8(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr in range(max(0, cr - 1), min(h, cr + 2)):
                    for nc in range(max(0, cc - 1), min(w, cc + 2)):
                        if (nr == cr and nc == cc) or seen[nr, nc] or int(grid[nr, nc]) == bg:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr})
    return components


def _scan_key(coords: np.ndarray) -> tuple[int, int]:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    r, c = coords[order[0]]
    return int(r), int(c)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr == r and nc == c) or not bool(mask[nr, nc]) or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_is_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return len(coords) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _parse_fragments(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = get_bg(grid)
    components = _find_components8(grid, bg)
    assert components, "expected at least one non-background fragment"

    color_hits: dict[int, int] = defaultdict(int)
    singleton_hits: dict[int, int] = defaultdict(int)
    color_counts_by_component: list[dict[int, int]] = []

    for comp in components:
        coords = comp["coords"]
        values = grid[coords[:, 0], coords[:, 1]]
        uniq, counts = np.unique(values, return_counts=True)
        counts_map = {int(color): int(count) for color, count in zip(uniq, counts)}
        assert len(counts_map) == 2, "each fragment must contain one body color plus one anchor color"
        color_counts_by_component.append(counts_map)
        for color, count in counts_map.items():
            color_hits[color] += 1
            if count == 1:
                singleton_hits[color] += 1

    anchor_candidates = [
        color
        for color, hit_count in color_hits.items()
        if hit_count == len(components) and singleton_hits.get(color, 0) == len(components)
    ]
    assert len(anchor_candidates) == 1, "expected a unique shared anchor color"
    anchor_color = int(anchor_candidates[0])

    fragments: list[dict[str, object]] = []
    for comp, counts_map in zip(components, color_counts_by_component):
        coords = comp["coords"]
        body_colors = [color for color in counts_map if color != anchor_color]
        assert len(body_colors) == 1, "each fragment must have exactly one body color"
        body_color = int(body_colors[0])

        anchor_pixels = coords[grid[coords[:, 0], coords[:, 1]] == anchor_color]
        assert anchor_pixels.shape[0] == 1, "each fragment must have exactly one anchor pixel"
        anchor = tuple(map(int, anchor_pixels[0]))

        rel = {
            (int(r) - anchor[0], int(c) - anchor[1])
            for r, c in coords
        }
        fragments.append(
            {
                "body_color": body_color,
                "anchor": anchor,
                "rel_coords": rel,
                "scan_key": _scan_key(coords),
            }
        )
    return bg, fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments by their shared body color. For each pair, align
    the two fragments on their grey anchors; their union must be a solid
    rectangle. Place that rectangle at the earlier fragment's anchor position and
    erase the later fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, fragments = _parse_fragments(g)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for frag in fragments:
        by_color[int(frag["body_color"])].append(frag)

    out = np.full_like(g, bg)
    for body_color, pair in by_color.items():
        assert len(pair) == 2, "each body color must appear in exactly two fragments"
        pair.sort(key=lambda frag: frag["scan_key"])
        first, second = pair
        anchor_r, anchor_c = first["anchor"]

        union = set(first["rel_coords"]) | set(second["rel_coords"])
        r0 = min(r for r, _ in union)
        r1 = max(r for r, _ in union)
        c0 = min(c for _, c in union)
        c1 = max(c for _, c in union)
        assert len(union) == (r1 - r0 + 1) * (c1 - c0 + 1), "aligned fragments must complete a solid rectangle"

        rr0 = anchor_r + r0
        rr1 = anchor_r + r1
        cc0 = anchor_c + c0
        cc1 = anchor_c + c1
        assert 0 <= rr0 <= rr1 < g.shape[0]
        assert 0 <= cc0 <= cc1 < g.shape[1]
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(body_color)

    return out


BG_COLOR = 0
ANCHOR_COLOR = 5
BODY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PAIR_COUNT_RANGE = (2, 4)
GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
RECT_HEIGHT_RANGE = (3, 6)
RECT_WIDTH_RANGE = (3, 7)
PLACEMENT_PAD = 1
GENERATION_ATTEMPTS = 300


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _profile(length: int, anchor_idx: int, anchor_value: int, lo: int, hi: int) -> list[int]:
    profile = [0] * length
    profile[anchor_idx] = anchor_value
    for i in range(anchor_idx - 1, -1, -1):
        left = max(lo, profile[i + 1] - 1)
        right = min(hi, profile[i + 1] + 1)
        profile[i] = random.randint(left, right)
    for i in range(anchor_idx + 1, length):
        left = max(lo, profile[i - 1] - 1)
        right = min(hi, profile[i - 1] + 1)
        profile[i] = random.randint(left, right)
    return profile


def _make_pair_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    for _ in range(100):
        if random.random() < 0.5:
            anchor_r = random.randint(0, h - 1)
            anchor_c = random.randint(1, w - 2)
            cuts = _profile(h, anchor_r, anchor_c, 1, w - 2)
            first = np.zeros((h, w), dtype=bool)
            second = np.zeros((h, w), dtype=bool)
            for r, cut in enumerate(cuts):
                first[r, : cut + 1] = True
                second[r, cut + 1 :] = True
            second[anchor_r, anchor_c] = True
            if len(set(cuts)) == 1 and random.random() < 0.75:
                continue
        else:
            anchor_r = random.randint(1, h - 2)
            anchor_c = random.randint(0, w - 1)
            cuts = _profile(w, anchor_c, anchor_r, 1, h - 2)
            first = np.zeros((h, w), dtype=bool)
            second = np.zeros((h, w), dtype=bool)
            for c, cut in enumerate(cuts):
                first[: cut + 1, c] = True
                second[cut + 1 :, c] = True
            second[anchor_r, anchor_c] = True
            if len(set(cuts)) == 1 and random.random() < 0.75:
                continue

        if not (bool(first[anchor_r, anchor_c]) and bool(second[anchor_r, anchor_c])):
            continue
        if not (_mask_connected(first) and _mask_connected(second)):
            continue
        if _mask_is_rectangle(first) and _mask_is_rectangle(second):
            continue
        if int(first.sum()) <= 2 or int(second.sum()) <= 2:
            continue
        return first, second, (anchor_r, anchor_c)
    return None


def _make_patch(mask: np.ndarray, anchor: tuple[int, int], body_color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    patch[mask] = int(body_color)
    patch[anchor] = ANCHOR_COLOR
    return patch


def _place_boxes(
    sizes: list[tuple[int, int]],
    row_start: int,
    row_stop: int,
    grid_w: int,
    pad: int,
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    used: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = sizes[idx]
        candidates = [
            (r, c)
            for r in range(row_start, row_stop - h + 1)
            for c in range(1, grid_w - w)
        ]
        random.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            box = (r, c, r + h - 1, c + w - 1)
            if all(_boxes_separate(box, other, pad) for other in used):
                chosen = box
                break
        if chosen is None:
            return None
        used.append(chosen)
        placed[idx] = chosen

    return [box for box in placed if box is not None]


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG_COLOR
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if height < 8 or width < 8:
        return None

    colors = random.sample(BODY_COLORS, pair_count)
    pair_data: list[dict[str, object]] = []
    sizes: list[tuple[int, int]] = []
    for color in colors:
        h = random.randint(*RECT_HEIGHT_RANGE)
        w = random.randint(*RECT_WIDTH_RANGE)
        masks = _make_pair_masks(h, w)
        if masks is None:
            return None
        first_mask, second_mask, anchor = masks
        pair_data.append(
            {
                "color": color,
                "anchor": anchor,
                "first_patch": _make_patch(first_mask, anchor, color),
                "second_patch": _make_patch(second_mask, anchor, color),
            }
        )
        sizes.append((h, w))

    mid = height // 2
    top_boxes = _place_boxes(sizes, 1, mid - 1, width, PLACEMENT_PAD)
    bottom_boxes = _place_boxes(sizes, mid + 1, height - 1, width, PLACEMENT_PAD)
    if top_boxes is None or bottom_boxes is None:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    expected = np.full_like(grid, BG_COLOR)

    for data, first_box, second_box in zip(pair_data, top_boxes, bottom_boxes):
        fr0, fc0, fr1, fc1 = first_box
        sr0, sc0, _sr1, _sc1 = second_box
        _stamp(grid, data["first_patch"], fr0, fc0)
        _stamp(grid, data["second_patch"], sr0, sc0)
        expected[fr0 : fr1 + 1, fc0 : fc1 + 1] = int(data["color"])

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != BG_COLOR:
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-aligned fragment puzzle")
