import random
from collections import Counter

import numpy as np


CORNER_LABELS = ("TL", "TR", "BL", "BR")
QUADRANT_TO_BLOCK = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}

GEN_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    components: list[np.ndarray] = []
    starts = np.argwhere(mask)
    for sr, sc in starts:
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _quadrant_bounds(shape: tuple[int, int], quadrant: str) -> tuple[int, int, int, int]:
    h, w = shape
    split_r = h // 2
    split_c = w // 2
    if quadrant == "TL":
        return 0, 0, split_r - 1, split_c - 1
    if quadrant == "TR":
        return 0, split_c, split_r - 1, w - 1
    if quadrant == "BL":
        return split_r, 0, h - 1, split_c - 1
    if quadrant == "BR":
        return split_r, split_c, h - 1, w - 1
    raise ValueError(f"unknown quadrant: {quadrant}")


def _closest_corner(center_r: float, center_c: float, bounds: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bounds
    options = {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BL": (r1, c0),
        "BR": (r1, c1),
    }
    scored = sorted(
        ((label, (center_r - rr) ** 2 + (center_c - cc) ** 2) for label, (rr, cc) in options.items()),
        key=lambda item: item[1],
    )
    assert scored[0][1] < scored[1][1], "marker corner bias must be unambiguous"
    return scored[0][0]


def _component_center(coords: np.ndarray) -> tuple[float, float]:
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _detect_panel_and_sprite(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = _component_coords(grid != bg, connectivity=8)
    assert len(comps) == 2, "expected exactly two foreground objects"

    panel_crop = None
    sprite_crop = None
    for coords in comps:
        bbox = bbox_coords(coords)
        crop = crop_bbox(grid, bbox)
        colors = sorted(set(map(int, crop[crop != bg].ravel())))
        full = bool(np.all(crop != bg))
        if full and len(colors) >= 2:
            assert panel_crop is None, "expected a single solid multicolor panel"
            panel_crop = crop
        else:
            assert sprite_crop is None, "expected a single sprite object"
            sprite_crop = crop

    assert panel_crop is not None and sprite_crop is not None
    assert len(set(map(int, sprite_crop[sprite_crop != bg].ravel()))) == 1, "sprite must be monochrome"
    return panel_crop, sprite_crop


def _panel_markers(panel: np.ndarray) -> list[dict]:
    colors = Counter(map(int, panel.ravel()))
    panel_color, _count = colors.most_common(1)[0]
    marker_mask = panel != int(panel_color)
    parts = _component_coords(marker_mask, connectivity=4)
    assert parts, "panel must contain at least one marker"

    markers = []
    seen_quadrants = set()
    for coords in parts:
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each marker must be 2x2"
        vals = panel[coords[:, 0], coords[:, 1]]
        marker_color = int(vals[0])
        assert np.all(vals == marker_color), "each marker must be monochrome"

        center_r, center_c = _component_center(coords)
        quadrant = ("T" if center_r < panel.shape[0] / 2.0 else "B") + ("L" if center_c < panel.shape[1] / 2.0 else "R")
        bounds = _quadrant_bounds(panel.shape, quadrant)
        bias = _closest_corner(center_r, center_c, bounds)
        assert quadrant not in seen_quadrants, "expected at most one marker per quadrant"
        seen_quadrants.add(quadrant)
        markers.append({"quadrant": quadrant, "bias": bias, "color": marker_color})
    return markers


def _sprite_heavy_corner(sprite: np.ndarray, bg: int) -> str:
    mask = sprite != bg
    coords = np.argwhere(mask)
    assert coords.size > 0, "sprite must be non-empty"
    h, w = sprite.shape
    row_score = int(np.sum(2 * coords[:, 0] - (h - 1)))
    col_score = int(np.sum(2 * coords[:, 1] - (w - 1)))
    assert row_score != 0 and col_score != 0, "sprite center of mass must prefer one corner"
    return ("T" if row_score < 0 else "B") + ("L" if col_score < 0 else "R")


def _sprite_heavy_corner_or_none(mask: np.ndarray) -> str | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    h, w = mask.shape
    row_score = int(np.sum(2 * coords[:, 0] - (h - 1)))
    col_score = int(np.sum(2 * coords[:, 1] - (w - 1)))
    if row_score == 0 or col_score == 0:
        return None
    return ("T" if row_score < 0 else "B") + ("L" if col_score < 0 else "R")


def _orient_sprite(sprite: np.ndarray, src_corner: str, dst_corner: str) -> np.ndarray:
    out = sprite.copy()
    if src_corner[0] != dst_corner[0]:
        out = np.flipud(out)
    if src_corner[1] != dst_corner[1]:
        out = np.fliplr(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the panel markers into a 2x2 array of recolored sprite copies with marker-driven orientation.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of one monochrome sprite and one separate solid rectangular panel.
    - The panel contains one or more 2x2 monochrome markers, with at most one marker in each quadrant.

    Rule:
    - Crop the sprite to its bounding box and use that size as the block size of a 2x2 output canvas.
    - For each marked panel quadrant, place one copy of the sprite in the corresponding output block.
    - Recolor that copy to the marker color and choose among identity, left-right mirror, up-down mirror, and 180-degree rotation so the sprite's heavy corner matches the marker's corner bias inside its quadrant.

    Key insight:
    - Each 2x2 marker encodes three visible signals at once: its panel quadrant chooses the destination block, its local offset chooses the sprite orientation, and its own color chooses the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    panel, sprite = _detect_panel_and_sprite(g, bg)
    markers = _panel_markers(panel)

    heavy_corner = _sprite_heavy_corner(sprite, bg)
    sh, sw = sprite.shape
    out = np.full((2 * sh, 2 * sw), int(bg), dtype=int)

    for marker in markers:
        transformed = _orient_sprite(sprite, heavy_corner, marker["bias"])
        patch = np.full_like(transformed, int(bg))
        patch[transformed != bg] = int(marker["color"])
        br, bc = QUADRANT_TO_BLOCK[marker["quadrant"]]
        r0 = br * sh
        c0 = bc * sw
        out[r0 : r0 + sh, c0 : c0 + sw] = patch
    return out


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _random_distinct_colors(count: int) -> list[int]:
    return random.sample(list(range(10)), count)


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask, connectivity=4)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _gen_tl_heavy_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = np.zeros((h, w), dtype=bool)
        seeds = {(0, 0), (0, min(1, w - 1)), (min(1, h - 1), 0)}
        for cell in seeds:
            mask[cell] = True

        bottom_col = random.randint(0, max(0, w // 2))
        right_row = random.randint(0, max(0, h // 2))

        for r in range(h):
            if r <= h - 1:
                mask[r, bottom_col] = True
        for c in range(w):
            if c <= w - 1:
                mask[right_row, c] = True

        target_area = random.randint(max(5, (h * w) // 3), min(h * w - 1, h * w - 2))
        while int(mask.sum()) < target_area:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.append((rr, cc))
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            frontier.sort(key=lambda cell: (cell[0] + cell[1], abs(cell[0] - right_row) + abs(cell[1] - bottom_col), random.random()))
            pick_count = min(len(frontier), random.randint(1, 3))
            rr, cc = random.choice(frontier[:pick_count])
            mask[rr, cc] = True

        if not _mask_connected(mask):
            continue
        if not _touches_all_sides(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if _sprite_heavy_corner_or_none(mask) != "TL":
            continue
        return mask
    raise RuntimeError("failed to build a suitable sprite mask")


def _transform_mask_to_corner(mask: np.ndarray, corner: str) -> np.ndarray:
    out = mask.copy()
    if corner[0] == "B":
        out = np.flipud(out)
    if corner[1] == "R":
        out = np.fliplr(out)
    return out


def _marker_position_options(qh: int, qw: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "TL": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "TR": [(0, qw - 2), (0, qw - 3), (1, qw - 2), (1, qw - 3)],
        "BL": [(qh - 2, 0), (qh - 3, 0), (qh - 2, 1), (qh - 3, 1)],
        "BR": [(qh - 2, qw - 2), (qh - 3, qw - 2), (qh - 2, qw - 3), (qh - 3, qw - 3)],
    }


def _build_panel(panel_color: int, marker_specs: list[tuple[str, str, int]], qh: int, qw: int) -> np.ndarray:
    panel = np.full((2 * qh, 2 * qw), int(panel_color), dtype=int)
    local_options = _marker_position_options(qh, qw)
    quadrant_origin = {
        "TL": (0, 0),
        "TR": (0, qw),
        "BL": (qh, 0),
        "BR": (qh, qw),
    }
    for quadrant, bias, color in marker_specs:
        base_r, base_c = quadrant_origin[quadrant]
        dr, dc = random.choice(local_options[bias])
        panel[base_r + dr : base_r + dr + 2, base_c + dc : base_c + dc + 2] = int(color)
    return panel


def _render_input(
    bg: int,
    sprite: np.ndarray,
    panel: np.ndarray,
) -> np.ndarray | None:
    sh, sw = sprite.shape
    ph, pw = panel.shape
    fg_area = int(np.count_nonzero(sprite != bg) + np.count_nonzero(panel != bg))

    for _ in range(120):
        h = random.randint(max(ph, sh) + 4, 30)
        w = random.randint(max(pw, sw) + 4, 30)
        if h * w <= 2 * fg_area + 12:
            continue
        grid = np.full((h, w), int(bg), dtype=int)

        pr = random.randint(1, h - ph - 1)
        pc = random.randint(1, w - pw - 1)
        sr = random.randint(1, h - sh - 1)
        sc = random.randint(1, w - sw - 1)

        panel_box = (pr, pc, pr + ph - 1, pc + pw - 1)
        sprite_box = (sr, sc, sr + sh - 1, sc + sw - 1)
        if not _boxes_separate(panel_box, sprite_box, padding=1):
            continue

        grid[pr : pr + ph, pc : pc + pw] = panel
        sprite_mask = sprite != bg
        grid[sr : sr + sh, sc : sc + sw][sprite_mask] = sprite[sprite_mask]
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.choices([1, 2, 3, 4], weights=[1, 4, 4, 3])[0]
    palette = _random_distinct_colors(marker_count + 3)
    bg, panel_color, sprite_color = palette[:3]
    marker_colors = palette[3:]

    sprite_corner = random.choice(CORNER_LABELS)
    sprite_mask = _transform_mask_to_corner(_gen_tl_heavy_mask(), sprite_corner)
    sprite = np.full(sprite_mask.shape, int(bg), dtype=int)
    sprite[sprite_mask] = int(sprite_color)

    qh = random.randint(5, 7)
    qw = random.randint(5, 7)
    chosen_quadrants = random.sample(list(CORNER_LABELS), marker_count)
    chosen_biases = random.sample(list(CORNER_LABELS), marker_count)
    marker_specs = [
        (quadrant, bias, color)
        for quadrant, bias, color in zip(chosen_quadrants, chosen_biases, marker_colors)
    ]
    panel = _build_panel(panel_color, marker_specs, qh, qw)

    grid = _render_input(bg, sprite, panel)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if get_bg(grid) != bg:
        return None
    if len(_component_coords(grid != bg, connectivity=8)) != 2:
        return None
    if _sprite_heavy_corner(sprite, bg) != sprite_corner:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
