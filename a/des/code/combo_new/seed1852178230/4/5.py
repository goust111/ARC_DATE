from __future__ import annotations

import random
from collections import deque
from itertools import product

import numpy as np


BG = 0
YELLOW = 4
ARROW = 9
BASE_COLORS = (1, 2, 3, 5, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_OFFSETS = {
    "up": ((-1, 0), (1, -1), (1, 1)),
    "down": ((1, 0), (-1, -1), (-1, 1)),
    "left": ((0, -1), (-1, 1), (1, 1)),
    "right": ((0, 1), (-1, -1), (-1, 1)),
}

GRID_MIN = 14
GRID_MAX = 22
RECT_SIDE_MIN = 5
RECT_SIDE_MAX_2 = 8
RECT_SIDE_MAX_3 = 7
MAX_OVERLAP_AREA = 5
CLIP_MIN_OVERLAP = 3
SCENE_ATTEMPTS = 700
PLACEMENT_ATTEMPTS = 180


def _scan_colors(grid: np.ndarray, excluded: set[int]) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in np.asarray(grid).ravel():
        color = int(value)
        if color in excluded or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = DIR4 if connectivity == 4 else DIR8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _inside_rect(cell: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _intersection_rect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    inter = _intersection_rect(a, b)
    return 0 if inter is None else _rect_area(inter)


def _contains_rect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _expanded_intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ar0 -= margin
    ac0 -= margin
    ar1 += margin
    ac1 += margin
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _arrow_pixels(center: tuple[int, int], direction: str) -> tuple[tuple[int, int], ...]:
    r, c = center
    return tuple((int(r + dr), int(c + dc)) for dr, dc in ARROW_OFFSETS[direction])


def _match_arrow(coords: np.ndarray) -> dict[str, object] | None:
    pts = {tuple(map(int, cell)) for cell in coords}
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    for center_r in range(min(rs) - 1, max(rs) + 2):
        for center_c in range(min(cs) - 1, max(cs) + 2):
            center = (int(center_r), int(center_c))
            for direction in DIR_STEPS:
                if set(_arrow_pixels(center, direction)) == pts:
                    return {
                        "center": center,
                        "direction": direction,
                        "pixels": tuple(sorted(pts)),
                    }
    return None


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    maroon = {tuple(map(int, cell)) for cell in np.argwhere(g == ARROW)}
    arrows_by_pixels: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}

    for center_r in range(1, h - 1):
        for center_c in range(1, w - 1):
            zone = {
                (center_r + dr, center_c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }
            maroons_in_zone = zone & maroon
            if len(maroons_in_zone) != 3:
                continue
            for direction in DIR_STEPS:
                pixels = tuple(sorted(_arrow_pixels((center_r, center_c), direction)))
                if set(pixels) == maroons_in_zone:
                    arrows_by_pixels[pixels] = {
                        "center": (int(center_r), int(center_c)),
                        "direction": direction,
                        "pixels": pixels,
                    }

    covered = {pixel for pixels in arrows_by_pixels for pixel in pixels}
    if covered != maroon:
        raise AssertionError("all maroon pixels must belong to exactly one arrow")
    return [arrows_by_pixels[key] for key in sorted(arrows_by_pixels)]


def _arrow_fit_score(
    grid: np.ndarray, arrow: dict[str, object], rect: tuple[int, int, int, int], color: int
) -> int:
    score = 0
    center_r, center_c = arrow["center"]  # type: ignore[index]
    for dr, dc in product((-1, 0, 1), repeat=2):
        if dr == 0 and dc == 0:
            continue
        rr = center_r + dr
        cc = center_c + dc
        if _inside_rect((int(rr), int(cc)), rect) and int(grid[rr, cc]) == color:
            score += 1
    return score


def _infer_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = _scan_colors(g, excluded={BG, ARROW})
    blocks: list[dict[str, object]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        rect = _bbox(coords)
        blocks.append(
            {
                "color": int(color),
                "rect": rect,
                "visible_count": int(coords.shape[0]),
                "visible_coords": coords,
            }
        )

    arrows = _find_arrows(g)
    if len(arrows) != len(blocks):
        raise AssertionError("expected one visible arrow per rectangle")

    taken: set[int] = set()
    for arrow in arrows:
        candidates: list[tuple[int, int]] = []
        for idx, block in enumerate(blocks):
            rect = block["rect"]  # type: ignore[index]
            if all(_inside_rect(pixel, rect) for pixel in arrow["pixels"]):  # type: ignore[index]
                score = _arrow_fit_score(g, arrow, rect, int(block["color"]))
                candidates.append((score, idx))
        if not candidates:
            raise AssertionError("arrow does not belong to any reconstructed rectangle")
        candidates.sort(key=lambda item: (-item[0], tuple(blocks[item[1]]["rect"])))  # type: ignore[index]
        owner = next((idx for _score, idx in candidates if idx not in taken), None)
        if owner is None:
            raise AssertionError("could not assign arrows uniquely")
        taken.add(int(owner))
        blocks[owner]["direction"] = arrow["direction"]
        blocks[owner]["arrow_pixels"] = arrow["pixels"]
        blocks[owner]["arrow_center"] = arrow["center"]

    for block in blocks:
        rect = block["rect"]  # type: ignore[index]
        visible_count = int(block["visible_count"]) + len(block["arrow_pixels"])  # type: ignore[index]
        k = _rect_area(rect) - visible_count
        if k < 0:
            raise AssertionError("visible area cannot exceed rectangle area")
        block["k"] = int(k)
    return blocks


def _infer_draw_order(blocks: list[dict[str, object]]) -> list[int]:
    order_keys = {
        int(block["color"]): tuple(block["rect"])  # type: ignore[index]
        for block in blocks
    }
    by_color = {int(block["color"]): block for block in blocks}
    edges: dict[int, set[int]] = {int(block["color"]): set() for block in blocks}
    indeg: dict[int, int] = {int(block["color"]): 0 for block in blocks}

    for a in blocks:
        ar0, ac0, ar1, ac1 = a["rect"]  # type: ignore[index]
        a_color = int(a["color"])
        for b in blocks:
            b_color = int(b["color"])
            if a_color == b_color:
                continue
            coords = b["visible_coords"]  # type: ignore[index]
            inside = (
                (coords[:, 0] >= ar0)
                & (coords[:, 0] <= ar1)
                & (coords[:, 1] >= ac0)
                & (coords[:, 1] <= ac1)
            )
            if bool(np.any(inside)) and b_color not in edges[a_color]:
                edges[a_color].add(b_color)
                indeg[b_color] += 1

    ready = sorted((color for color, deg in indeg.items() if deg == 0), key=lambda color: order_keys[color])
    out: list[int] = []
    while ready:
        color = ready.pop(0)
        out.append(int(color))
        for nxt in sorted(edges[color], key=lambda c: order_keys[c]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda c: order_keys[c])
    if len(out) != len(blocks):
        return [int(block["color"]) for block in sorted(blocks, key=lambda block: tuple(block["rect"]))]  # type: ignore[index]
    return out


def _translate_rect(
    rect: tuple[int, int, int, int], direction: str, steps: int
) -> tuple[int, int, int, int]:
    dr, dc = DIR_STEPS[direction]
    return (
        int(rect[0] + dr * steps),
        int(rect[1] + dc * steps),
        int(rect[2] + dr * steps),
        int(rect[3] + dc * steps),
    )


def _clip_rect(
    rect: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0 = max(0, int(rect[0]))
    c0 = max(0, int(rect[1]))
    r1 = min(h - 1, int(rect[2]))
    c1 = min(w - 1, int(rect[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _adjacency_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    touch_a = np.zeros_like(mask_a, dtype=bool)
    touch_b = np.zeros_like(mask_b, dtype=bool)

    down = mask_a[:-1, :] & mask_b[1:, :]
    touch_a[:-1, :] |= down
    touch_b[1:, :] |= down

    up = mask_a[1:, :] & mask_b[:-1, :]
    touch_a[1:, :] |= up
    touch_b[:-1, :] |= up

    right = mask_a[:, :-1] & mask_b[:, 1:]
    touch_a[:, :-1] |= right
    touch_b[:, 1:] |= right

    left = mask_a[:, 1:] & mask_b[:, :-1]
    touch_a[:, 1:] |= left
    touch_b[:, :-1] |= left

    return touch_a, touch_b


def _simulate_output(
    blocks: list[dict[str, object]], shape: tuple[int, int]
) -> tuple[np.ndarray, list[dict[str, object]], bool]:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    moved: list[dict[str, object]] = []
    for block in blocks:
        color = int(block["color"])
        rect = block["rect"]  # type: ignore[index]
        direction = str(block["direction"])
        k = int(block["k"])
        moved_rect = _translate_rect(rect, direction, k)
        clipped = _clip_rect(moved_rect, shape)
        mask = np.zeros((h, w), dtype=bool)
        if clipped is not None:
            r0, c0, r1, c1 = clipped
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            out[mask] = color
        moved.append(
            {
                "color": color,
                "mask": mask,
                "moved_rect": moved_rect,
                "clipped_rect": clipped,
                "clipped": clipped != moved_rect,
            }
        )

    yellow_mask = np.zeros((h, w), dtype=bool)
    has_adj = False
    for i in range(len(moved)):
        for j in range(i + 1, len(moved)):
            touch_i, touch_j = _adjacency_masks(moved[i]["mask"], moved[j]["mask"])  # type: ignore[index]
            if bool(np.any(touch_i) or np.any(touch_j)):
                has_adj = True
            yellow_mask |= touch_i | touch_j
    out[yellow_mask] = YELLOW
    return out, moved, has_adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each reconstructed rectangle by its occlusion count, then fuse newly touching borders in yellow.

    Input structure:
    - The background is black.
    - Each foreground object is a solid-color axis-aligned rectangle with a unique color, and later rectangles may partially cover earlier ones.
    - Every rectangle contains a maroon 3-pixel arrowhead that points up, down, left, or right.

    Rule:
    - Recover each rectangle's full bounding box from its visible color footprint and assign its internal arrow to get the motion direction.
    - Let K be the number of rectangle cells hidden by later rectangles, i.e. full rectangle area minus the rectangle's visible own-color cells and its visible arrow cells.
    - Translate the full rectangle by K cells along the arrow direction, clip anything that exits the grid, erase all arrows, and recolor every pair of 4-neighbor boundary cells between distinct moved rectangles to yellow.

    Key insight:
    - The visible color fragments are only partial evidence: the missing parts come from the rectangle bbox, while the arrow contributes to the visibility count but not to the final shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    blocks = _infer_blocks(g)
    order = _infer_draw_order(blocks)
    block_by_color = {int(block["color"]): block for block in blocks}
    ordered_blocks = [block_by_color[color] for color in order]
    out, _moved, _has_adj = _simulate_output(ordered_blocks, g.shape)
    return out


def _sample_sizes(rng: random.Random, n_blocks: int) -> tuple[int, int]:
    side_max = RECT_SIDE_MAX_2 if n_blocks == 2 else RECT_SIDE_MAX_3
    return rng.randint(RECT_SIDE_MIN, side_max), rng.randint(RECT_SIDE_MIN, side_max)


def _all_positions(height: int, width: int, h: int, w: int) -> list[tuple[int, int, int, int]]:
    return [
        (r0, c0, r0 + h - 1, c0 + w - 1)
        for r0 in range(height - h + 1)
        for c0 in range(width - w + 1)
    ]


def _edge_distance(rect: tuple[int, int, int, int], shape: tuple[int, int], direction: str) -> int:
    h, w = shape
    r0, c0, r1, c1 = rect
    if direction == "up":
        return int(r0)
    if direction == "down":
        return int(h - 1 - r1)
    if direction == "left":
        return int(c0)
    return int(w - 1 - c1)


def _boundary_visible_after_overlap(
    target: tuple[int, int, int, int], cover: tuple[int, int, int, int]
) -> bool:
    inter = _intersection_rect(target, cover)
    if inter is None:
        return True
    tr0, tc0, tr1, tc1 = target
    ir0, ic0, ir1, ic1 = inter
    top_hidden = ir0 == tr0 and ic0 <= tc0 and ic1 >= tc1
    bottom_hidden = ir1 == tr1 and ic0 <= tc0 and ic1 >= tc1
    left_hidden = ic0 == tc0 and ir0 <= tr0 and ir1 >= tr1
    right_hidden = ic1 == tc1 and ir0 <= tr0 and ir1 >= tr1
    return not (top_hidden or bottom_hidden or left_hidden or right_hidden)


def _candidate_seed_rects(
    shape: tuple[int, int], h: int, w: int, direction: str, bias_edge: bool
) -> list[tuple[int, int, int, int]]:
    rects = _all_positions(shape[0], shape[1], h, w)
    if not bias_edge:
        return rects
    return [
        rect
        for rect in rects
        if _edge_distance(rect, shape, direction) <= 2
    ]


def _candidate_disjoint_rects(
    shape: tuple[int, int],
    h: int,
    w: int,
    existing: list[dict[str, object]],
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for rect in _all_positions(shape[0], shape[1], h, w):
        ok = True
        for block in existing:
            other = block["rect"]  # type: ignore[index]
            if _intersection_area(rect, other) > 0 or _expanded_intersects(rect, other, margin=1):
                ok = False
                break
        if ok:
            out.append(rect)
    return out


def _candidate_overlap_rects(
    shape: tuple[int, int],
    h: int,
    w: int,
    target: tuple[int, int, int, int],
    blockers: list[dict[str, object]],
    min_area: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for rect in _all_positions(shape[0], shape[1], h, w):
        area = _intersection_area(rect, target)
        if not (min_area <= area <= MAX_OVERLAP_AREA):
            continue
        if _contains_rect(target, rect) or _contains_rect(rect, target):
            continue
        if not _boundary_visible_after_overlap(target, rect):
            continue
        bad = False
        for block in blockers:
            other = block["rect"]  # type: ignore[index]
            if _intersection_area(rect, other) > 0 or _expanded_intersects(rect, other, margin=1):
                bad = True
                break
        if not bad:
            out.append(rect)
    return out


def _exclusive_mask(blocks: list[dict[str, object]], idx: int, shape: tuple[int, int]) -> np.ndarray:
    mask = _rect_mask(shape, blocks[idx]["rect"])  # type: ignore[index]
    for j, block in enumerate(blocks):
        if j == idx:
            continue
        mask &= ~_rect_mask(shape, block["rect"])  # type: ignore[index]
    return mask


def _assign_arrows(blocks: list[dict[str, object]], shape: tuple[int, int], rng: random.Random) -> bool:
    candidate_lists: list[list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]] = []
    for idx, block in enumerate(blocks):
        rect = block["rect"]  # type: ignore[index]
        direction = str(block["direction"])
        ex = _exclusive_mask(blocks, idx, shape)
        r0, c0, r1, c1 = rect
        candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
        for center_r in range(r0 + 1, r1):
            for center_c in range(c0 + 1, c1):
                zone = [(center_r + dr, center_c + dc) for dr, dc in product((-1, 0, 1), repeat=2)]
                if not all(ex[rr, cc] for rr, cc in zone):
                    continue
                pixels = _arrow_pixels((center_r, center_c), direction)
                candidates.append(((int(center_r), int(center_c)), pixels))
        if not candidates:
            return False
        rng.shuffle(candidates)
        candidate_lists.append(candidates)

    chosen_pixels: list[tuple[tuple[int, int], ...]] = [tuple()] * len(blocks)
    for idx in sorted(range(len(blocks)), key=lambda i: len(candidate_lists[i])):
        placed = False
        for center, pixels in candidate_lists[idx]:
            conflict = False
            for other_pixels in chosen_pixels:
                if not other_pixels:
                    continue
                for r, c in pixels:
                    for rr, cc in other_pixels:
                        if max(abs(r - rr), abs(c - cc)) <= 1:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if not conflict:
                blocks[idx]["arrow_center"] = center
                blocks[idx]["arrow_pixels"] = pixels
                chosen_pixels[idx] = pixels
                placed = True
                break
        if not placed:
            return False
    return True


def _render_input(blocks: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for block in blocks:
        r0, c0, r1, c1 = block["rect"]  # type: ignore[index]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(block["color"])
        for rr, cc in block["arrow_pixels"]:  # type: ignore[index]
            grid[rr, cc] = ARROW
    return grid


def _visible_bbox_ok(grid: np.ndarray, block: dict[str, object]) -> bool:
    color = int(block["color"])
    rect = block["rect"]  # type: ignore[index]
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return False
    if _bbox(coords) != rect:
        return False
    r0, c0, r1, c1 = rect
    if not np.any(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.any(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.any(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.any(grid[r0 : r1 + 1, c1] == color):
        return False
    return True


def _pairwise_output_overlap(moved: list[dict[str, object]]) -> bool:
    for i in range(len(moved)):
        for j in range(i + 1, len(moved)):
            if bool(np.any(moved[i]["mask"] & moved[j]["mask"])):  # type: ignore[index]
                return True
    return False


def _choose_overlap_pattern(
    n_blocks: int,
    want_adjacency: bool,
    want_clipping: bool,
    rng: random.Random,
) -> tuple[bool, ...]:
    if n_blocks == 2:
        patterns = [(False,), (True,)]
    else:
        patterns = [(False, False), (True, False), (False, True), (True, True)]
    if want_adjacency or want_clipping:
        patterns = [pattern for pattern in patterns if any(pattern)]
    return rng.choice(patterns)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    shape = (height, width)
    n_blocks = rng.choice((2, 3))
    want_adjacency = rng.random() < 0.5
    want_clipping = rng.random() < 0.35
    overlap_flags = _choose_overlap_pattern(n_blocks, want_adjacency, want_clipping, rng)
    colors = rng.sample(BASE_COLORS, n_blocks)
    directions = [rng.choice(tuple(DIR_STEPS)) for _ in range(n_blocks)]
    clip_target = None
    if want_clipping and any(overlap_flags):
        clip_target = rng.choice([idx for idx, flag in enumerate(overlap_flags) if flag])

    blocks: list[dict[str, object]] = []
    for idx in range(n_blocks):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            h, w = _sample_sizes(rng, n_blocks)
            if idx == 0:
                rects = _candidate_seed_rects(
                    shape,
                    h,
                    w,
                    directions[idx],
                    bias_edge=(clip_target == 0),
                )
            elif overlap_flags[idx - 1]:
                min_area = CLIP_MIN_OVERLAP if clip_target == idx - 1 else 1
                rects = _candidate_overlap_rects(
                    shape,
                    h,
                    w,
                    blocks[idx - 1]["rect"],  # type: ignore[index]
                    blocks[:-1],
                    min_area=min_area,
                )
            else:
                rects = _candidate_disjoint_rects(shape, h, w, blocks)
            if not rects:
                continue
            rect = rng.choice(rects)
            blocks.append(
                {
                    "color": int(colors[idx]),
                    "rect": rect,
                    "direction": directions[idx],
                }
            )
            placed = True
            break
        if not placed:
            return None

    for idx in range(1, n_blocks):
        inter = _intersection_area(blocks[idx]["rect"], blocks[idx - 1]["rect"])  # type: ignore[index]
        if overlap_flags[idx - 1]:
            if not (1 <= inter <= MAX_OVERLAP_AREA):
                return None
        elif inter != 0:
            return None
        for earlier in range(idx - 1):
            if _intersection_area(blocks[idx]["rect"], blocks[earlier]["rect"]) != 0:  # type: ignore[index]
                return None
            if _expanded_intersects(blocks[idx]["rect"], blocks[earlier]["rect"], margin=1):  # type: ignore[index]
                return None

    if not _assign_arrows(blocks, shape, rng):
        return None
    grid = _render_input(blocks, shape)

    for block in blocks:
        if not _visible_bbox_ok(grid, block):
            return None
        for rr, cc in block["arrow_pixels"]:  # type: ignore[index]
            if int(grid[rr, cc]) != ARROW:
                return None

    arrows = _find_arrows(grid)
    if len(arrows) != n_blocks:
        return None

    k_values = []
    for idx, block in enumerate(blocks):
        if idx < n_blocks - 1 and overlap_flags[idx]:
            k = _intersection_area(block["rect"], blocks[idx + 1]["rect"])  # type: ignore[index]
        else:
            k = 0
        block["k"] = int(k)
        k_values.append(int(k))

    if any(flag for flag in overlap_flags) and not any(k_values):
        return None
    if not any(overlap_flags) and any(k_values):
        return None

    expected, moved, has_adj = _simulate_output(blocks, shape)
    if _pairwise_output_overlap(moved):
        return None
    if any(meta["clipped_rect"] is None for meta in moved):
        return None

    has_clipping = any(bool(meta["clipped"]) for meta in moved)
    if has_clipping != want_clipping:
        return None
    if has_adj != want_adjacency:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
