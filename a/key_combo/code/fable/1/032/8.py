from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, b: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, b)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if int(connectivity) == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    out: list[dict] = []
    for color in colors_present(g, exclude={int(bg)}):
        for coords in components_mask(g == int(color), connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _coords_key(coords: np.ndarray) -> tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _tight_mask_from_component(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return local


def _touches_grid_edge(component: dict, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h, w = map(int, shape)
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _touches_bbox_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _anchor_hole_info(component: dict) -> tuple[BBox, np.ndarray] | None:
    local = _tight_mask_from_component(component)
    h, w = local.shape
    if h < 3 or w < 3:
        return None
    if not (local[0, :].any() and local[-1, :].any() and local[:, 0].any() and local[:, -1].any()):
        return None

    hole = ~local
    if not hole.any() or not _touches_bbox_edge(hole):
        return None
    if len(components_mask(hole, connectivity=4)) != 1:
        return None

    hole_bbox = bbox(hole)
    assert hole_bbox is not None
    return hole_bbox, crop_bbox(hole, hole_bbox)


def _select_anchor(components: list[dict], grid_shape: tuple[int, int]) -> tuple[dict, BBox, np.ndarray]:
    candidates: list[tuple[dict, BBox, np.ndarray]] = []
    for component in components:
        info = _anchor_hole_info(component)
        if info is None:
            continue
        hole_bbox_local, hole_mask = info
        r0, c0, _r1, _c1 = component["bbox"]
        hr0, hc0, hr1, hc1 = hole_bbox_local
        hole_bbox_global = (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1)
        candidates.append((component, hole_bbox_global, hole_mask))

    assert candidates, "expected an anchor component"
    return max(
        candidates,
        key=lambda item: (item[0]["area"], int(not _touches_grid_edge(item[0], grid_shape))),
    )


def _matching_scale(fragment_mask: np.ndarray, hole_mask: np.ndarray) -> int | None:
    fh, fw = fragment_mask.shape
    hh, hw = hole_mask.shape
    if fh > hh or fw > hw or hh % fh != 0 or hw % fw != 0:
        return None
    scale_h = hh // fh
    scale_w = hw // fw
    if scale_h != scale_w or scale_h < 2:
        return None
    scaled = scale_sprite(fragment_mask.astype(int), scale_h).astype(bool)
    if np.array_equal(scaled, hole_mask):
        return int(scale_h)
    return None


def _find_matches(
    components: list[dict],
    grid_shape: tuple[int, int],
    anchor_component: dict,
    hole_mask: np.ndarray,
) -> list[tuple[dict, int]]:
    anchor_key = _coords_key(anchor_component["coords"])
    color = int(anchor_component["color"])
    matches: list[tuple[dict, int]] = []
    for component in components:
        if int(component["color"]) != color:
            continue
        if _coords_key(component["coords"]) == anchor_key:
            continue
        if not _touches_grid_edge(component, grid_shape):
            continue
        scale = _matching_scale(_tight_mask_from_component(component), hole_mask)
        if scale is not None:
            matches.append((component, scale))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique edge fragment whose integer upscale exactly fills the anchor notch and complete the rectangle.

    Input structure:
    - The background is the dominant color.
    - One large monochrome anchor object is a solid rectangle with one boundary-connected notch removed.
    - Several smaller same-colored fragments touch the grid edge, and exactly one of them matches the notch after an integer pixel upscale.

    Rule:
    - Detect the anchor by finding the large component whose bounding box is a full rectangle except for one connected missing region.
    - Crop that missing notch, then test every same-colored edge fragment for the unique integer scale factor k>=2 whose upscaled mask equals the notch.
    - Erase the matching fragment from the edge and paint its upscaled copy into the notch, leaving all distractor fragments unchanged.

    Key insight:
    - The needed fit is not a same-size translation; the unique solution appears only when fragment shape and integer scale are considered together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4)
    anchor_component, hole_bbox, hole_mask = _select_anchor(components, g.shape)
    matches = _find_matches(components, g.shape, anchor_component, hole_mask)
    assert len(matches) == 1, "expected exactly one fragment-scale match"
    fragment_component, _scale = matches[0]

    out = g.copy()
    fragment_coords = fragment_component["coords"]
    out[fragment_coords[:, 0], fragment_coords[:, 1]] = int(bg)

    hr0, hc0, hr1, hc1 = hole_bbox
    hole_view = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
    hole_view[hole_mask] = int(anchor_component["color"])
    return out


GEN_ATTEMPTS = 500
GEN_FRAGMENT_TOTAL_RANGE = (3, 6)
GEN_TARGET_SCALE_CHOICES = (2, 2, 3, 3, 4)
GEN_BG_COLORS = tuple(range(10))
GEN_TARGET_MARGIN = 2
GEN_RECT_MARGIN_RANGE = (0, 5)
GEN_OPPOSITE_MARGIN_RANGE = (1, 6)
GEN_FRAGMENT_PAD = 1
GEN_ANCHOR_PAD = 2


def _random_connected_mask(
    *,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_area: int = 3,
    allow_full: bool = False,
    min_dim: int = 1,
) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        max_area = h * w if allow_full else h * w - 1
        if max_area < min_area:
            continue
        area = random.randint(min_area, max_area)
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < area:
            frontier: list[Coord] = []
            seen = set()
            for r, c in np.argwhere(mask):
                for nr, nc in neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if int(mask.sum()) != area:
            continue
        b = bbox(mask)
        assert b is not None
        mask = crop_bbox(mask, b)
        if min(mask.shape) < min_dim:
            continue
        if not allow_full and bool(mask.all()):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        return mask
    return None


def _mutate_mask(mask: np.ndarray) -> np.ndarray | None:
    for _ in range(80):
        padded = np.pad(mask, 1, constant_values=False)
        rr = random.randrange(padded.shape[0])
        cc = random.randrange(padded.shape[1])
        candidate = padded.copy()
        candidate[rr, cc] = ~candidate[rr, cc]
        if not candidate.any():
            continue
        b = bbox(candidate)
        assert b is not None
        candidate = crop_bbox(candidate, b)
        if candidate.sum() < 2:
            continue
        if len(components_mask(candidate, connectivity=4)) != 1:
            continue
        return candidate
    return None


def _build_anchor_from_notch(notch_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], str] | None:
    nh, nw = notch_mask.shape
    for _ in range(120):
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            top = 0
            bottom = random.randint(*GEN_OPPOSITE_MARGIN_RANGE)
            left = random.randint(*GEN_RECT_MARGIN_RANGE)
            right = random.randint(*GEN_RECT_MARGIN_RANGE)
            if left + right == 0:
                continue
        elif side == "bottom":
            top = random.randint(*GEN_OPPOSITE_MARGIN_RANGE)
            bottom = 0
            left = random.randint(*GEN_RECT_MARGIN_RANGE)
            right = random.randint(*GEN_RECT_MARGIN_RANGE)
            if left + right == 0:
                continue
        elif side == "left":
            top = random.randint(*GEN_RECT_MARGIN_RANGE)
            bottom = random.randint(*GEN_RECT_MARGIN_RANGE)
            left = 0
            right = random.randint(*GEN_OPPOSITE_MARGIN_RANGE)
            if top + bottom == 0:
                continue
        else:
            top = random.randint(*GEN_RECT_MARGIN_RANGE)
            bottom = random.randint(*GEN_RECT_MARGIN_RANGE)
            left = random.randint(*GEN_OPPOSITE_MARGIN_RANGE)
            right = 0
            if top + bottom == 0:
                continue

        anchor = np.ones((top + nh + bottom, left + nw + right), dtype=bool)
        anchor[top : top + nh, left : left + nw][notch_mask] = False
        if len(components_mask(anchor, connectivity=4)) != 1:
            continue
        if not (anchor[0, :].any() and anchor[-1, :].any() and anchor[:, 0].any() and anchor[:, -1].any()):
            continue
        hole = ~anchor
        if len(components_mask(hole, connectivity=4)) != 1:
            continue
        hole_bbox = bbox(hole)
        assert hole_bbox is not None
        if not np.array_equal(crop_bbox(hole, hole_bbox), notch_mask):
            continue
        return anchor, (top, left), side
    return None


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _box_is_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)
    return coords + np.array([r0, c0], dtype=int)


def _edge_positions(grid_shape: tuple[int, int], mask_shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = map(int, grid_shape)
    mh, mw = map(int, mask_shape)
    positions: list[tuple[int, int]] = []
    if side == "top" and mw <= w:
        positions = [(0, c) for c in range(w - mw + 1)]
    elif side == "bottom" and mw <= w:
        positions = [(h - mh, c) for c in range(w - mw + 1)]
    elif side == "left" and mh <= h:
        positions = [(r, 0) for r in range(h - mh + 1)]
    elif side == "right" and mh <= h:
        positions = [(r, w - mw) for r in range(h - mh + 1)]
    random.shuffle(positions)
    return positions


def _place_edge_fragment(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> np.ndarray | None:
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        for r0, c0 in _edge_positions(grid.shape, mask.shape, side):
            if not _box_is_free(reserved, r0, c0, mask.shape[0], mask.shape[1], GEN_FRAGMENT_PAD):
                continue
            coords = _paint_mask(grid, mask, r0, c0, color)
            _reserve_box(reserved, r0, c0, mask.shape[0], mask.shape[1], GEN_FRAGMENT_PAD)
            return coords
    return None


def _target_mask_for_scale(scale: int) -> np.ndarray | None:
    if scale >= 4:
        max_dim = 3
    elif scale == 3:
        max_dim = 4
    else:
        max_dim = 5
    return _random_connected_mask(
        min_h=2,
        max_h=max_dim,
        min_w=2,
        max_w=max_dim,
        min_area=3,
        allow_full=False,
        min_dim=2,
    )


def _random_distractor(target_mask: np.ndarray, hole_mask: np.ndarray) -> np.ndarray | None:
    th, tw = target_mask.shape
    for _ in range(120):
        mode = random.choice(("mutate", "same_bbox", "random"))
        if mode == "mutate":
            candidate = _mutate_mask(target_mask)
        elif mode == "same_bbox":
            candidate = _random_connected_mask(
                min_h=th,
                max_h=th,
                min_w=tw,
                max_w=tw,
                min_area=2,
                allow_full=True,
                min_dim=1,
            )
            if candidate is not None and candidate.shape != target_mask.shape:
                candidate = None
        else:
            candidate = _random_connected_mask(
                min_h=1,
                max_h=max(4, th + 1),
                min_w=1,
                max_w=max(4, tw + 1),
                min_area=2,
                allow_full=True,
                min_dim=1,
            )
        if candidate is None:
            continue
        if np.array_equal(candidate, target_mask):
            continue
        if _matching_scale(candidate, hole_mask) is not None:
            continue
        return candidate
    return None


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice(GEN_BG_COLORS)
    fg = random.choice([color for color in GEN_BG_COLORS if color != bg])

    scale = random.choice(GEN_TARGET_SCALE_CHOICES)
    target_mask = _target_mask_for_scale(scale)
    if target_mask is None:
        return None
    notch_mask = scale_sprite(target_mask.astype(int), scale).astype(bool)

    anchor_build = _build_anchor_from_notch(notch_mask)
    if anchor_build is None:
        return None
    anchor_mask, notch_offset, _side = anchor_build
    ah, aw = anchor_mask.shape
    nh, nw = notch_mask.shape

    min_h = max(ah + 2 * GEN_TARGET_MARGIN, 16)
    min_w = max(aw + 2 * GEN_TARGET_MARGIN, 16)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    if grid_h - ah < 2 * GEN_TARGET_MARGIN or grid_w - aw < 2 * GEN_TARGET_MARGIN:
        return None

    anchor_r = random.randint(GEN_TARGET_MARGIN, grid_h - ah - GEN_TARGET_MARGIN)
    anchor_c = random.randint(GEN_TARGET_MARGIN, grid_w - aw - GEN_TARGET_MARGIN)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    anchor_coords = np.argwhere(anchor_mask) + np.array([anchor_r, anchor_c], dtype=int)
    grid[anchor_coords[:, 0], anchor_coords[:, 1]] = int(fg)
    _reserve_box(reserved, anchor_r, anchor_c, ah, aw, GEN_ANCHOR_PAD)

    fragment_total = random.randint(*GEN_FRAGMENT_TOTAL_RANGE)
    fragments: list[dict] = [{"mask": target_mask, "target": True}]
    while len(fragments) < fragment_total:
        distractor = _random_distractor(target_mask, notch_mask)
        if distractor is None:
            return None
        fragments.append({"mask": distractor, "target": False})
    random.shuffle(fragments)

    target_coords: np.ndarray | None = None
    for fragment in fragments:
        coords = _place_edge_fragment(grid, reserved, fragment["mask"], fg)
        if coords is None:
            return None
        if fragment["target"]:
            target_coords = coords
    assert target_coords is not None

    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None

    expected = grid.copy()
    expected[target_coords[:, 0], target_coords[:, 1]] = int(bg)
    hole_r = anchor_r + notch_offset[0]
    hole_c = anchor_c + notch_offset[1]
    expected[hole_r : hole_r + nh, hole_c : hole_c + nw][notch_mask] = int(fg)

    bg_found = get_bg(grid)
    if bg_found != bg:
        return None
    components = find_components(grid, bg=bg_found, connectivity=4)
    anchor_component, hole_bbox, hole_mask = _select_anchor(components, grid.shape)
    matches = _find_matches(components, grid.shape, anchor_component, hole_mask)
    if len(matches) != 1:
        return None
    matched_component, matched_scale = matches[0]
    if matched_scale != scale:
        return None
    if _coords_key(matched_component["coords"]) != _coords_key(target_coords):
        return None
    if hole_bbox != (hole_r, hole_c, hole_r + nh - 1, hole_c + nw - 1):
        return None
    if not np.array_equal(hole_mask, notch_mask):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-scale ARC input")
