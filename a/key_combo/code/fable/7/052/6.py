import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
TEMPLATE_COLORS = (1, 3, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def apply_transform(tile: np.ndarray, op: str) -> np.ndarray:
    if op == "id":
        return tile.copy()
    if op == "lr":
        return np.fliplr(tile)
    if op == "ud":
        return np.flipud(tile)
    if op == "rot180":
        return np.rot90(tile, 2)
    raise ValueError(f"unknown transform: {op}")


def corner_pos(name: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, w - 1
    if name == "bl":
        return h - 1, 0
    if name == "br":
        return h - 1, w - 1
    raise ValueError(f"unknown corner: {name}")


def detect_corner_name(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    named = {
        "tl": bool(mask[0, 0]),
        "tr": bool(mask[0, w - 1]),
        "bl": bool(mask[h - 1, 0]),
        "br": bool(mask[h - 1, w - 1]),
    }
    hits = [name for name, flag in named.items() if flag]
    return hits[0] if len(hits) == 1 else None


def transformed_yellow_corner(tile: np.ndarray, op: str) -> str:
    transformed = apply_transform(tile, op)
    name = detect_corner_name(transformed == YELLOW)
    assert name is not None, "transformed template must keep one yellow corner"
    return name


def analyze_instance(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int, int, int, str]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg)
    assert len(comps) >= 2, "expected one template plus at least one target"

    targets: list[tuple[int, int, int, int, str]] = []
    template_candidates: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []

    for comp in comps:
        box = bbox_coords(comp)
        r0, c0, r1, c1 = box
        tile = crop_bbox(g, box)
        h, w = tile.shape
        colors = set(map(int, np.unique(tile)))
        area = int(comp.shape[0])

        if (
            area == h * w
            and colors.issubset({GREY, RED})
            and int(np.count_nonzero(tile == RED)) == 1
            and int(np.count_nonzero(tile == GREY)) == h * w - 1
        ):
            corner_name = detect_corner_name(tile == RED)
            if corner_name is not None:
                targets.append((r0, c0, r1, c1, corner_name))
                continue

        if h != w and GREY not in colors and int(np.count_nonzero(tile == YELLOW)) == 1:
            yellow_corner = detect_corner_name(tile == YELLOW)
            if yellow_corner is not None:
                template_candidates.append((tile, box))

    assert targets, "expected target rectangles"
    dims = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1, _ in targets}
    assert len(dims) == 1, "all targets must have the same shape"
    target_shape = next(iter(dims))

    matching_templates = [
        (tile, box) for tile, box in template_candidates if tile.shape == target_shape
    ]
    assert len(matching_templates) == 1, "expected one template matching the target size"
    template, template_box = matching_templates[0]
    return template, template_box, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the template into each grey rectangle using the red corner as a pose marker.

    Input structure:
    - A black background contains one connected non-square multicolor template.
    - Exactly one template corner cell is yellow, and that yellow cell is part of the template.
    - One or more solid grey rectangles of the same size also appear; each has one corner grey cell replaced by a red pixel.

    Rule:
    - Treat the template bbox as the stamp.
    - For each grey rectangle, choose the unique transform from {identity, left-right mirror, up-down mirror, 180-degree rotation} whose yellow corner lands on the rectangle's red corner.
    - Overwrite the whole rectangle with that transformed template, so the yellow corner covers the red marker; keep the background and original template unchanged.

    Key insight:
    - The single red corner on each target is the image of the template's single yellow corner, so corner correspondence alone fixes the stamp orientation.
    """
    g = np.asarray(grid, dtype=int)
    template, _template_box, targets = analyze_instance(g)

    op_by_corner = {
        transformed_yellow_corner(template, op): op
        for op in ("id", "lr", "ud", "rot180")
    }
    assert len(op_by_corner) == 4, "yellow-corner mapping must distinguish all four transforms"

    out = g.copy()
    for r0, c0, r1, c1, red_corner in targets:
        patch = apply_transform(template, op_by_corner[red_corner])
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def build_template(h: int, w: int) -> np.ndarray:
    palette = random.sample(TEMPLATE_COLORS, 3)
    a, b, c = palette

    tile = np.full((h, w), a, dtype=int)
    style = random.choice(("row_frontier", "col_frontier", "diag_split"))
    rr, cc = np.indices((h, w))

    if style == "row_frontier":
        cuts = np.array([random.randint(1, w - 2) for _ in range(h)], dtype=int)[:, None]
        tile = np.where(cc < cuts, a, b).astype(int)
    elif style == "col_frontier":
        cuts = np.array([random.randint(1, h - 2) for _ in range(w)], dtype=int)[None, :]
        tile = np.where(rr < cuts, a, b).astype(int)
    else:
        bias = random.randint(1, h + w - 3)
        tile = np.where(rr + cc <= bias, a, b).astype(int)

    anchor_r = random.randint(1, h - 2)
    anchor_c = random.randint(1, w - 2)
    tile[anchor_r, anchor_c] = c
    extra_cells = [(anchor_r, anchor_c + 1), (anchor_r + 1, anchor_c)]
    random.shuffle(extra_cells)
    for rr0, cc0 in extra_cells[: random.randint(1, 2)]:
        if rr0 < h and cc0 < w:
            tile[rr0, cc0] = c

    for _ in range(3):
        rr0 = random.randint(0, h - 1)
        cc0 = random.randint(0, w - 1)
        tile[rr0, cc0] = random.choice((a, b, c))

    yellow_corner = random.choice(CORNERS)
    yr, yc = corner_pos(yellow_corner, (h, w))
    tile[yr, yc] = YELLOW
    return tile


def template_has_pose_signal(tile: np.ndarray) -> bool:
    stripped = tile.copy()
    stripped[stripped == YELLOW] = BLACK
    variants = {
        apply_transform(stripped, op).tobytes() for op in ("id", "lr", "ud", "rot180")
    }
    return len(variants) == 4 and len(set(map(int, np.unique(stripped))) - {BLACK}) >= 2


def fits(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def place_boxes(
    grid_shape: tuple[int, int], box_shape: tuple[int, int], count: int
) -> list[tuple[int, int, int, int]] | None:
    gh, gw = grid_shape
    h, w = box_shape
    coords = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    random.shuffle(coords)
    placed: list[tuple[int, int, int, int]] = []
    for r0, c0 in coords:
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if fits(box, placed, gap=1):
            placed.append(box)
            if len(placed) == count:
                return placed
    return None


def render_target(shape: tuple[int, int], red_corner: str) -> np.ndarray:
    patch = np.full(shape, GREY, dtype=int)
    rr, cc = corner_pos(red_corner, shape)
    patch[rr, cc] = RED
    return patch


def validate_generated(grid: np.ndarray) -> None:
    template, _template_box, targets = analyze_instance(grid)
    assert template_has_pose_signal(template), "template must need the full transform, not only yellow relocation"

    red_corners = [corner for *_box, corner in targets]
    assert len(set(red_corners)) >= 2, "need at least two distinct target corners"
    assert any(corner != detect_corner_name(template == YELLOW) for corner in red_corners), (
        "need at least one non-identity target"
    )

    out = main(grid)
    op_by_corner = {
        transformed_yellow_corner(template, op): op
        for op in ("id", "lr", "ud", "rot180")
    }
    for r0, c0, r1, c1, corner in targets:
        expected = apply_transform(template, op_by_corner[corner])
        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(3, 6)
        w = random.randint(4, 7)
        if h == w:
            continue

        template = build_template(h, w)
        if not template_has_pose_signal(template):
            continue

        target_count = random.randint(2, 4)
        distinct_count = min(target_count, random.randint(2, 4))
        chosen_corners = random.sample(CORNERS, distinct_count)
        while len(chosen_corners) < target_count:
            chosen_corners.append(random.choice(chosen_corners))
        random.shuffle(chosen_corners)

        yellow_corner = detect_corner_name(template == YELLOW)
        assert yellow_corner is not None
        if len(set(chosen_corners)) < 2 or all(corner == yellow_corner for corner in chosen_corners):
            continue

        min_h = max(12, h + 2)
        min_w = max(12, w + 2)
        max_h = min(30, max(min_h, (target_count + 1) * h + target_count + 3))
        max_w = min(30, max(min_w, (target_count + 1) * w + target_count + 3))
        if min_h > max_h or min_w > max_w:
            continue
        gh = random.randint(min_h, max_h)
        gw = random.randint(min_w, max_w)

        boxes = place_boxes((gh, gw), (h, w), target_count + 1)
        if boxes is None:
            continue

        grid = np.full((gh, gw), BLACK, dtype=int)
        random.shuffle(boxes)
        template_box = boxes[0]
        r0, c0, r1, c1 = template_box
        grid[r0 : r1 + 1, c0 : c1 + 1] = template

        for box, red_corner in zip(boxes[1:], chosen_corners):
            tr0, tc0, tr1, tc1 = box
            grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = render_target((h, w), red_corner)

        try:
            validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid angle-stamp instance")
