from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8)
POSITIVE_SIDES = {"top", "right"}
NEGATIVE_SIDES = {"bottom", "left"}
GEN_RETRIES = 400


def _neighbors(r: int, c: int, connectivity: int):
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _find_components(
    grid: np.ndarray, *, bg: int = BG, connectivity: int = 8
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _relative_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _detect_notch_side(component: dict) -> str:
    mask = _relative_mask(component)
    h, w = mask.shape
    candidates: list[str] = []

    for c in range(1, w - 1):
        if not mask[0, c] and mask[0, c - 1] and mask[0, c + 1]:
            candidates.append("top")
        if not mask[h - 1, c] and mask[h - 1, c - 1] and mask[h - 1, c + 1]:
            candidates.append("bottom")
    for r in range(1, h - 1):
        if not mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]:
            candidates.append("left")
        if not mask[r, w - 1] and mask[r - 1, w - 1] and mask[r + 1, w - 1]:
            candidates.append("right")

    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one outer notch, got {candidates}")
    return candidates[0]


def _build_ring(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.zeros((side, side), dtype=int)
    for depth, color in enumerate(colors):
        lo = depth
        hi = side - depth - 1
        if lo == hi:
            out[lo, hi] = color
            continue
        out[lo, lo : hi + 1] = color
        out[hi, lo : hi + 1] = color
        out[lo : hi + 1, lo] = color
        out[lo : hi + 1, hi] = color
    return out


def _parse_groups(grid: np.ndarray) -> list[dict]:
    components = _find_components(grid, bg=BG, connectivity=8)
    if not components:
        raise AssertionError("expected colored layers")

    n = len(components)
    parent = [None] * n
    for i in range(n):
        child_box = components[i]["bbox"]
        candidates = [
            j
            for j in range(n)
            if _strictly_contains(components[j]["bbox"], child_box)
        ]
        if candidates:
            parent[i] = min(candidates, key=lambda j: _bbox_area(components[j]["bbox"]))

    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)

    groups: list[dict] = []
    for root in range(n):
        if parent[root] is not None:
            continue
        chain = []
        cur = root
        while True:
            chain.append(components[cur])
            if not children[cur]:
                break
            if len(children[cur]) != 1:
                raise AssertionError("nesting must be a single chain per group")
            cur = children[cur][0]

        layers = sorted(chain, key=lambda comp: _bbox_area(comp["bbox"]), reverse=True)
        for a, b in zip(layers, layers[1:]):
            if not _strictly_contains(a["bbox"], b["bbox"]):
                raise AssertionError("layer bboxes must be strictly nested")

        groups.append(
            {
                "layers": layers,
                "outer_bbox": layers[0]["bbox"],
                "notch_side": _detect_notch_side(layers[0]),
            }
        )

    groups.sort(key=lambda group: group["outer_bbox"][1])
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each nested-shape group into a square-ring summary whose color order is flipped by the outer notch direction.

    Input structure:
    - Black (0) is the background.
    - The grid contains 2-3 separated groups of monochromatic 8-connected nested layers.
    - Each group has 3-4 distinct-colored layers with strictly nested bounding boxes, and only the outer layer has a single black notch on one bbox side.

    Rule:
    - Detect each group's nesting chain by bounding-box containment and order its layers from outermost to innermost.
    - Read the outer notch side: top/right keeps that color order, while bottom/left reverses it.
    - Convert each group into a square summary of side 2*layers-1, using one square ring per color from outer to inner.
    - Sort summaries by the input groups' left bbox coordinate, place them in one row with a one-cell black gap, and top-align them.

    Key insight:
    - The nested hierarchy supplies the color sequence, but the outer-layer notch decides whether that sequence is encoded forward or backward in the compressed ring summary.
    """
    g = np.asarray(grid, dtype=int)
    groups = _parse_groups(g)

    summaries = []
    for group in groups:
        colors = [layer["color"] for layer in group["layers"]]
        if group["notch_side"] in NEGATIVE_SIDES:
            colors = list(reversed(colors))
        summaries.append(_build_ring(colors))

    out_h = max(summary.shape[0] for summary in summaries)
    out_w = sum(summary.shape[1] for summary in summaries) + len(summaries) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c = 0
    for summary in summaries:
        h, w = summary.shape
        out[:h, c : c + w] = summary
        c += w + 1
    return out


def _corner_cut(mask: np.ndarray, corner: str, size: int) -> None:
    h, w = mask.shape
    if size == 1:
        if corner == "tl":
            mask[0, 0] = False
        elif corner == "tr":
            mask[0, w - 1] = False
        elif corner == "bl":
            mask[h - 1, 0] = False
        else:
            mask[h - 1, w - 1] = False
        return

    if h < 3 or w < 3:
        return
    if corner == "tl":
        mask[0, 0] = False
        mask[0, 1] = False
        mask[1, 0] = False
    elif corner == "tr":
        mask[0, w - 1] = False
        mask[0, w - 2] = False
        mask[1, w - 1] = False
    elif corner == "bl":
        mask[h - 1, 0] = False
        mask[h - 2, 0] = False
        mask[h - 1, 1] = False
    else:
        mask[h - 1, w - 1] = False
        mask[h - 2, w - 1] = False
        mask[h - 1, w - 2] = False


def _shape_mask(h: int, w: int, *, irregular: bool) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if not irregular:
        return mask

    corners = ["tl", "tr", "bl", "br"]
    n_cuts = random.randint(1, 2 if min(h, w) >= 4 else 1)
    for corner in random.sample(corners, n_cuts):
        _corner_cut(mask, corner, size=random.choice((1, 2)))
    return mask


def _build_nested_boxes(n_layers: int) -> list[tuple[int, int, int, int]]:
    for _ in range(40):
        outer_h = random.randint(2 * n_layers + 1, 2 * n_layers + 4)
        outer_w = random.randint(2 * n_layers + 1, 2 * n_layers + 4)
        boxes = [(0, 0, outer_h - 1, outer_w - 1)]

        asymmetry = False
        valid = True
        for depth in range(1, n_layers):
            r0, c0, r1, c1 = boxes[-1]
            cur_h = r1 - r0 + 1
            cur_w = c1 - c0 + 1
            remain = n_layers - depth
            min_h = 3 + 2 * (remain - 1)
            min_w = 3 + 2 * (remain - 1)
            choices = []
            for top in (1, 2):
                for bottom in (1, 2):
                    if cur_h - top - bottom < min_h:
                        continue
                    for left in (1, 2):
                        for right in (1, 2):
                            if cur_w - left - right < min_w:
                                continue
                            choices.append((top, bottom, left, right))
            if not choices:
                valid = False
                break
            top, bottom, left, right = random.choice(choices)
            asymmetry |= top != bottom or left != right
            boxes.append((r0 + top, c0 + left, r1 - bottom, c1 - right))

        if valid and asymmetry:
            return boxes
    raise AssertionError("failed to build asymmetric nested boxes")


def _make_group_patch(
    n_layers: int, colors: list[int], notch_side: str
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    boxes = _build_nested_boxes(n_layers)
    outer_h = boxes[0][2] - boxes[0][0] + 1
    outer_w = boxes[0][3] - boxes[0][1] + 1
    patch = np.zeros((outer_h, outer_w), dtype=int)

    for idx, (color, box) in enumerate(zip(colors, boxes)):
        r0, c0, r1, c1 = box
        mask = _shape_mask(r1 - r0 + 1, c1 - c0 + 1, irregular=idx > 0)
        view = patch[r0 : r1 + 1, c0 : c1 + 1]
        view[mask] = color

    if notch_side == "top":
        notch = (0, random.randint(1, outer_w - 2))
    elif notch_side == "bottom":
        notch = (outer_h - 1, random.randint(1, outer_w - 2))
    elif notch_side == "left":
        notch = (random.randint(1, outer_h - 2), 0)
    else:
        notch = (random.randint(1, outer_h - 2), outer_w - 1)
    patch[notch] = BG
    return patch, boxes


def _ring_signature(group: dict) -> tuple[tuple[int, ...], str]:
    colors = tuple(layer["color"] for layer in group["layers"])
    if group["notch_side"] in NEGATIVE_SIDES:
        colors = tuple(reversed(colors))
    return colors, group["notch_side"]


def _semantic_validate(grid: np.ndarray) -> None:
    groups = _parse_groups(grid)
    assert len(groups) in (2, 3)
    assert any(group["notch_side"] in POSITIVE_SIDES for group in groups)
    assert any(group["notch_side"] in NEGATIVE_SIDES for group in groups)

    seen_signatures = set()
    area_shortcut_broken = False
    for group in groups:
        layers = group["layers"]
        assert 3 <= len(layers) <= 4
        colors = [layer["color"] for layer in layers]
        assert len(colors) == len(set(colors))

        signature = _ring_signature(group)
        assert signature not in seen_signatures
        seen_signatures.add(signature)

        ordered = colors
        flipped = list(reversed(colors))
        assert ordered != flipped

        visible_areas = [layer["area"] for layer in layers]
        if any(a < b for a, b in zip(visible_areas, visible_areas[1:])):
            area_shortcut_broken = True

    assert area_shortcut_broken, "need at least one group where pixel area order is misleading"

    if len(groups) >= 2:
        by_col = [group["outer_bbox"] for group in groups]
        by_row = sorted(groups, key=lambda group: group["outer_bbox"][0])
        assert [g["outer_bbox"] for g in by_row] != by_col, "need column-order and row-order to differ"

    expected = []
    for group in groups:
        colors = [layer["color"] for layer in group["layers"]]
        if group["notch_side"] in NEGATIVE_SIDES:
            colors = list(reversed(colors))
        expected.append(_build_ring(colors))

    out = main(grid)
    out_h = max(arr.shape[0] for arr in expected)
    out_w = sum(arr.shape[1] for arr in expected) + len(expected) - 1
    rebuilt = np.zeros((out_h, out_w), dtype=int)
    c = 0
    for arr in expected:
        rebuilt[: arr.shape[0], c : c + arr.shape[1]] = arr
        c += arr.shape[1] + 1
    assert np.array_equal(out, rebuilt)


def _attempt_generate() -> np.ndarray | None:
    n_groups = random.choice((2, 2, 3))
    sides = [random.choice(tuple(POSITIVE_SIDES)), random.choice(tuple(NEGATIVE_SIDES))]
    while len(sides) < n_groups:
        sides.append(random.choice(("top", "right", "bottom", "left")))
    random.shuffle(sides)

    layer_counts = [random.choice((3, 4)) for _ in range(n_groups)]
    if n_groups == 3 and sum(1 for x in layer_counts if x == 4) == 3:
        layer_counts[random.randrange(n_groups)] = 3

    group_specs = []
    used_signatures = set()
    special_idx = random.randrange(n_groups)
    for i in range(n_groups):
        n_layers = layer_counts[i]
        chosen = None
        for _ in range(60):
            colors = random.sample(PALETTE, n_layers)
            try:
                patch, _ = _make_group_patch(n_layers, colors, sides[i])
                parsed = _parse_groups(patch)
            except AssertionError:
                continue
            if len(parsed) != 1:
                continue
            group = parsed[0]
            signature = _ring_signature(group)
            if signature in used_signatures:
                continue
            visible_areas = [layer["area"] for layer in group["layers"]]
            if i == special_idx and not any(a < b for a, b in zip(visible_areas, visible_areas[1:])):
                continue
            chosen = {"patch": patch, "group": group, "signature": signature}
            used_signatures.add(signature)
            break
        if chosen is None:
            return None
        group_specs.append(chosen)

    ordered_specs = random.sample(group_specs, len(group_specs))
    if n_groups == 2:
        left, right = ordered_specs
        left_h, left_w = left["patch"].shape
        right_h, right_w = right["patch"].shape
        min_h = left_h + right_h + 5
        min_w = max(left_w + 4, right_w + 4, 10)
        if min_h > 28 or min_w > 28:
            return None
        h = random.randint(max(22, min_h), 28)
        w = random.randint(max(22, min_w), 28)
        top_row = random.randint(1, 2)
        bottom_row = h - left_h - random.randint(1, 2)
        if bottom_row < top_row + right_h + 2:
            return None
        left_col_max = max(1, w - left_w - 4)
        left_col = random.randint(1, left_col_max)
        right_col = random.randint(min(w - right_w - 1, left_col + 2), w - right_w - 1)
        placements = [
            (left, bottom_row, left_col),
            (right, top_row, right_col),
        ]
    else:
        left, middle, right = ordered_specs
        left_h, left_w = left["patch"].shape
        mid_h, mid_w = middle["patch"].shape
        right_h, right_w = right["patch"].shape
        min_h = max(mid_h + max(left_h, right_h) + 5, 22)
        min_w = left_w + right_w + 5
        if min_h > 28 or min_w > 28:
            return None
        h = random.randint(min_h, 28)
        w = random.randint(max(22, min_w), 28)
        top_row = random.randint(1, 2)
        bottom_row = h - max(left_h, right_h) - random.randint(1, 2)
        if bottom_row < top_row + mid_h + 2:
            return None
        left_col = random.randint(1, max(1, w - (left_w + right_w + 4)))
        right_min = left_col + left_w + 2
        right_col = random.randint(right_min, w - right_w - 1)
        if right_col - left_col < 2:
            return None
        mid_low = left_col + 1
        mid_high = right_col - 1
        if mid_high > w - mid_w - 1:
            mid_high = w - mid_w - 1
        if mid_low > mid_high:
            return None
        mid_col = random.randint(mid_low, mid_high)
        placements = [
            (left, bottom_row, left_col),
            (middle, top_row, mid_col),
            (right, bottom_row, right_col),
        ]

    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    for spec, r0, c0 in placements:
        patch = spec["patch"]
        ph, pw = patch.shape
        if r0 < 1 or c0 < 1 or r0 + ph >= h or c0 + pw >= w:
            return None
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        for other in placed_boxes:
            if not (
                box[2] + 1 < other[0]
                or other[2] + 1 < box[0]
                or box[3] + 1 < other[1]
                or other[3] + 1 < box[1]
            ):
                return None
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        placed_boxes.append(box)

    try:
        _semantic_validate(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-notch ARC input")
