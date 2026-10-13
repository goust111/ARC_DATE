import random
from collections import Counter

import numpy as np


BLACK = 0
GEN_GREY = 5
GEN_RETRIES = 500
SIDES = ("top", "bottom", "left", "right")
CORNER_TO_AXES = {
    "TL": ("top", "left"),
    "TR": ("top", "right"),
    "BL": ("bottom", "left"),
    "BR": ("bottom", "right"),
}
AXES_TO_CORNER = {axes: corner for corner, axes in CORNER_TO_AXES.items()}
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _target_corner(side: str, corner: str) -> str:
    vert, horiz = CORNER_TO_AXES[corner]
    if side == "top":
        vert = "top"
    elif side == "bottom":
        vert = "bottom"
    elif side == "left":
        horiz = "left"
    elif side == "right":
        horiz = "right"
    else:
        raise ValueError(f"unknown side: {side}")
    return AXES_TO_CORNER[(vert, horiz)]


def _flip_flags(side: str, corner: str) -> tuple[bool, bool]:
    target = _target_corner(side, corner)
    orig_vert, orig_horiz = CORNER_TO_AXES[corner]
    target_vert, target_horiz = CORNER_TO_AXES[target]
    return orig_vert != target_vert, orig_horiz != target_horiz


def _corner_index(corner: str, height: int, width: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - 1
    if corner == "BL":
        return height - 1, 0
    if corner == "BR":
        return height - 1, width - 1
    raise ValueError(f"unknown corner: {corner}")


def _placement_bbox(anchor: tuple[int, int], target_corner: str, height: int, width: int) -> tuple[int, int, int, int]:
    ar, ac = anchor
    if target_corner == "TL":
        r0, c0 = ar, ac
    elif target_corner == "TR":
        r0, c0 = ar, ac - width + 1
    elif target_corner == "BL":
        r0, c0 = ar - height + 1, ac
    elif target_corner == "BR":
        r0, c0 = ar - height + 1, ac - width + 1
    else:
        raise ValueError(f"unknown corner: {target_corner}")
    return r0, c0, r0 + height - 1, c0 + width - 1


def _apply_flips(arr: np.ndarray, vflip: bool, hflip: bool) -> np.ndarray:
    out = np.asarray(arr, dtype=int)
    if vflip:
        out = np.flipud(out)
    if hflip:
        out = np.fliplr(out)
    return out


def _most_common(values: np.ndarray) -> int:
    flat = list(map(int, np.asarray(values).ravel()))
    if not flat:
        raise ValueError("cannot take mode of empty collection")
    return Counter(flat).most_common(1)[0][0]


def _infer_grey(grid: np.ndarray, anchor_colors: set[int]) -> int:
    h, w = grid.shape
    interior = grid[1 : h - 1, 1 : w - 1]
    vals = [
        int(v)
        for v in interior.ravel()
        if int(v) != BLACK and int(v) not in anchor_colors
    ]
    if not vals:
        raise ValueError("failed to infer grey rectangle color")
    return Counter(vals).most_common(1)[0][0]


def _anchor_map(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    anchors: dict[int, tuple[int, int]] = {}
    border_points = []
    for c in range(w):
        border_points.append((0, c))
        border_points.append((h - 1, c))
    for r in range(1, h - 1):
        border_points.append((r, 0))
        border_points.append((r, w - 1))

    for r, c in border_points:
        color = int(grid[r, c])
        if color == BLACK:
            continue
        if color in anchors:
            raise ValueError(f"duplicate anchor color on border: {color}")
        anchors[color] = (int(r), int(c))
    if not anchors:
        raise ValueError("expected at least one border anchor")
    return anchors


def _marker_corner(grid: np.ndarray, grey: int, r: int, c: int) -> str:
    up = int(grid[r - 1, c]) == grey
    down = int(grid[r + 1, c]) == grey
    left = int(grid[r, c - 1]) == grey
    right = int(grid[r, c + 1]) == grey
    key = (up, down, left, right)
    mapping = {
        (False, True, False, True): "TL",
        (False, True, True, False): "TR",
        (True, False, False, True): "BL",
        (True, False, True, False): "BR",
    }
    if key not in mapping:
        raise ValueError(f"invalid marker corner neighborhood at {(r, c)}: {key}")
    return mapping[key]


def _bbox_from_marker(grid: np.ndarray, grey: int, r: int, c: int, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    if corner == "TL":
        c1 = c
        while c1 + 1 < w and int(grid[r, c1 + 1]) == grey:
            c1 += 1
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, c]) == grey:
            r1 += 1
        return r, c, r1, c1
    if corner == "TR":
        c0 = c
        while c0 - 1 >= 0 and int(grid[r, c0 - 1]) == grey:
            c0 -= 1
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, c]) == grey:
            r1 += 1
        return r, c0, r1, c
    if corner == "BL":
        c1 = c
        while c1 + 1 < w and int(grid[r, c1 + 1]) == grey:
            c1 += 1
        r0 = r
        while r0 - 1 >= 0 and int(grid[r0 - 1, c]) == grey:
            r0 -= 1
        return r0, c, r, c1
    if corner == "BR":
        c0 = c
        while c0 - 1 >= 0 and int(grid[r, c0 - 1]) == grey:
            c0 -= 1
        r0 = r
        while r0 - 1 >= 0 and int(grid[r0 - 1, c]) == grey:
            r0 -= 1
        return r0, c0, r, c
    raise ValueError(f"unknown corner: {corner}")


def _validate_rectangle(sub: np.ndarray, grey: int, marker_color: int, marker_corner: str) -> None:
    height, width = sub.shape
    if height < 2 or width < 2:
        raise ValueError("rectangle is too small")
    mr, mc = _corner_index(marker_corner, height, width)
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    for r, c in np.argwhere(border):
        value = int(sub[r, c])
        if (int(r), int(c)) == (mr, mc):
            if value != marker_color:
                raise ValueError("marker corner color mismatch")
        elif value != grey:
            raise ValueError("rectangle border must remain grey")
    allowed = {BLACK, grey, marker_color}
    if not np.isin(sub, list(allowed)).all():
        raise ValueError("rectangle contains unexpected colors")


def _anchor_side(anchor: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = anchor
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    if on_top and on_left:
        return "top"
    if on_top and on_right:
        return "top"
    if on_bottom and on_left:
        return "bottom"
    if on_bottom and on_right:
        return "bottom"
    if on_top:
        return "top"
    if on_bottom:
        return "bottom"
    if on_left:
        return "left"
    if on_right:
        return "right"
    raise ValueError(f"anchor is not on the border: {anchor}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each grey marked rectangle to its color-matched border anchor with the needed mirror flip.

    Input structure:
    - Background is black.
    - Single-pixel colored anchors sit on the outer border; each anchor color is unique.
    - The interior contains filled grey rectangles with black asymmetric cutouts.
    - Every rectangle has exactly one non-grey colored corner pixel whose color matches one border anchor.

    Rule:
    - For each colored corner marker, identify which rectangle corner it occupies and recover that rectangle's full bounding box.
    - Find the border anchor with the same color.
    - Mirror the rectangle vertically and/or horizontally so that the marked corner becomes the corner that can touch that anchor from inside the grid, while preserving the marker's other visible orientation.
    - Paste the transformed rectangle so the marker overlaps the anchor, and remove all original central rectangles.

    Key insight:
    - The marker color selects the destination anchor, while the marker's corner position selects the mirror transform needed before placement on the border.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    h, w = g.shape

    anchors = _anchor_map(g)
    grey = _infer_grey(g, set(anchors))
    out = np.full_like(g, BLACK)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color == BLACK or color == grey or color not in anchors:
                continue

            marker_corner = _marker_corner(g, grey, r, c)
            r0, c0, r1, c1 = _bbox_from_marker(g, grey, r, c, marker_corner)
            sub = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            _validate_rectangle(sub, grey, color, marker_corner)

            anchor = anchors[color]
            side = _anchor_side(anchor, g.shape)
            target_corner = _target_corner(side, marker_corner)
            vflip, hflip = _flip_flags(side, marker_corner)
            transformed = _apply_flips(sub, vflip, hflip)

            pr0, pc0, pr1, pc1 = _placement_bbox(anchor, target_corner, transformed.shape[0], transformed.shape[1])
            if pr0 < 0 or pc0 < 0 or pr1 >= h or pc1 >= w:
                raise ValueError("placed rectangle would leave the grid")
            region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
            if np.any((region != BLACK) & (transformed != BLACK)):
                raise ValueError("placed rectangles overlap")
            out[pr0 : pr1 + 1, pc0 : pc1 + 1] = transformed

    return out


def _expanded_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 1) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_pattern(rng: random.Random, height: int, width: int) -> np.ndarray:
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    if len(interior) < 4:
        raise ValueError("rectangle interior too small for an asymmetric pattern")

    for _ in range(200):
        count = rng.randint(2, len(interior) - 1)
        chosen = rng.sample(interior, count)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        inner = mask[1:-1, 1:-1]
        if np.array_equal(inner, np.flipud(inner)):
            continue
        if np.array_equal(inner, np.fliplr(inner)):
            continue
        if np.array_equal(inner, np.flipud(np.fliplr(inner))):
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric pattern")


def _make_rectangle(rng: random.Random, height: int, width: int, color: int, corner: str) -> np.ndarray:
    rect = np.full((height, width), GEN_GREY, dtype=int)
    rect[_random_pattern(rng, height, width)] = BLACK
    mr, mc = _corner_index(corner, height, width)
    rect[mr, mc] = color
    return rect


def _choose_anchor(
    rng: random.Random,
    shape: tuple[int, int],
    occupied: np.ndarray,
    side: str,
    corner: str,
    height: int,
    width: int,
) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    h, w = shape
    target = _target_corner(side, corner)
    target_vert, target_horiz = CORNER_TO_AXES[target]

    if side in {"top", "bottom"}:
        row = 0 if side == "top" else h - 1
        col_lo = max(1, width - 1) if target_horiz == "right" else 1
        col_hi = min(w - 2, w - width) if target_horiz == "left" else w - 2
        candidates = list(range(col_lo, col_hi + 1))
        rng.shuffle(candidates)
        for col in candidates:
            anchor = (row, int(col))
            bbox = _placement_bbox(anchor, target, height, width)
            r0, c0, r1, c1 = bbox
            if 0 <= r0 and 0 <= c0 and r1 < h and c1 < w and _expanded_free(occupied, bbox, margin=1):
                return anchor, bbox
        return None

    col = 0 if side == "left" else w - 1
    row_lo = max(1, height - 1) if target_vert == "bottom" else 1
    row_hi = min(h - 2, h - height) if target_vert == "top" else h - 2
    candidates = list(range(row_lo, row_hi + 1))
    rng.shuffle(candidates)
    for row in candidates:
        anchor = (int(row), col)
        bbox = _placement_bbox(anchor, target, height, width)
        r0, c0, r1, c1 = bbox
        if 0 <= r0 and 0 <= c0 and r1 < h and c1 < w and _expanded_free(occupied, bbox, margin=1):
            return anchor, bbox
    return None


def _choose_center_bbox(
    rng: random.Random,
    shape: tuple[int, int],
    occupied: np.ndarray,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    row_start = 3
    row_stop = h - height - 3
    col_start = 3
    col_stop = w - width - 3
    if row_start > row_stop or col_start > col_stop:
        return None

    candidates = [(r, c) for r in range(row_start, row_stop + 1) for c in range(col_start, col_stop + 1)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        bbox = (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))
        if _expanded_free(occupied, bbox, margin=1):
            return bbox
    return None


def _build_expected(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for obj in objects:
        vflip, hflip = _flip_flags(obj["side"], obj["corner"])
        transformed = _apply_flips(obj["rect"], vflip, hflip)
        r0, c0, r1, c1 = obj["out_bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((region != BLACK) & (transformed != BLACK)):
            raise ValueError("generator created overlapping output rectangles")
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _has_branch_coverage(objects: list[dict]) -> bool:
    flips = [_flip_flags(obj["side"], obj["corner"]) for obj in objects]
    return (
        any(v for v, _ in flips)
        and any(h for _, h in flips)
        and any((not v) and (not h) for v, h in flips)
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(19, 29)
    w = rng.randint(19, 29)
    count = rng.randint(3, 4)
    sides = rng.sample(SIDES, count)
    colors = rng.sample(PALETTE, count)

    corners = [rng.choice(tuple(CORNER_TO_AXES)) for _ in range(count)]
    objects = [
        {
            "side": side,
            "color": color,
            "corner": corner,
            "height": rng.randint(4, 6),
            "width": rng.randint(4, 6),
        }
        for side, color, corner in zip(sides, colors, corners)
    ]
    if not _has_branch_coverage(objects):
        return None

    out_occ = np.zeros((h, w), dtype=bool)
    for obj in objects:
        choice = _choose_anchor(
            rng,
            (h, w),
            out_occ,
            obj["side"],
            obj["corner"],
            obj["height"],
            obj["width"],
        )
        if choice is None:
            return None
        anchor, bbox = choice
        obj["anchor"] = anchor
        obj["out_bbox"] = bbox
        obj["target_corner"] = _target_corner(obj["side"], obj["corner"])
        _mark_bbox(out_occ, bbox)

    center_occ = np.zeros((h, w), dtype=bool)
    for obj in objects:
        bbox = _choose_center_bbox(rng, (h, w), center_occ, obj["height"], obj["width"])
        if bbox is None:
            return None
        obj["in_bbox"] = bbox
        _mark_bbox(center_occ, bbox)

    grid = np.full((h, w), BLACK, dtype=int)
    for obj in objects:
        rect = _make_rectangle(rng, obj["height"], obj["width"], obj["color"], obj["corner"])
        obj["rect"] = rect
        ir0, ic0, ir1, ic1 = obj["in_bbox"]
        grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = rect
        ar, ac = obj["anchor"]
        if grid[ar, ac] != BLACK:
            return None
        grid[ar, ac] = obj["color"]

    expected = _build_expected((h, w), objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
