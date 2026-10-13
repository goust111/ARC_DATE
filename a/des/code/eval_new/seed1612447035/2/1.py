import random
from collections import Counter

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "right", "bottom", "left")
CW_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _rect_frame_info(coords: np.ndarray) -> dict | None:
    bbox, mask = _coords_to_mask(coords)
    h, w = mask.shape
    if h < 3 or w < 3:
        return None
    border = _rect_border_mask(h, w)
    if np.any(mask & ~border):
        return None
    missing = np.argwhere(border & ~mask)
    if len(missing) > 1:
        return None
    return {"bbox": bbox, "mask": mask, "missing": missing}


def _missing_side(info: dict) -> str:
    missing = info["missing"]
    assert len(missing) == 1
    mr, mc = map(int, missing[0])
    h, w = info["mask"].shape
    if mr == 0:
        assert 0 < mc < w - 1
        return "top"
    if mc == w - 1:
        assert 0 < mr < h - 1
        return "right"
    if mr == h - 1:
        assert 0 < mc < w - 1
        return "bottom"
    assert mc == 0 and 0 < mr < h - 1
    return "left"


def _find_display_panel(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            info = _rect_frame_info(coords)
            if info is None or len(info["missing"]) != 0:
                continue
            r0, c0, r1, c1 = info["bbox"]
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_mask = inner != bg
            inner_comps = _components_mask(inner_mask)
            if len(inner_comps) < 2:
                continue
            ok = True
            for comp in inner_comps:
                vals = np.unique(inner[comp[:, 0], comp[:, 1]])
                if len(vals) != 1:
                    ok = False
                    break
            if not ok:
                continue
            candidates.append((len(inner_comps), int(inner_mask.sum()), color, info))
    assert candidates, "expected one corner display panel"
    _, _, color, info = max(candidates, key=lambda item: (item[0], item[1]))
    info = dict(info)
    info["color"] = int(color)
    return info


def _extract_templates(grid: np.ndarray, panel: dict, bg: int) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = panel["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    templates: dict[int, np.ndarray] = {}
    for comp in _components_mask(inner != bg):
        vals = np.unique(inner[comp[:, 0], comp[:, 1]])
        assert len(vals) == 1
        color = int(vals[0])
        assert color not in templates
        (sr0, sc0, sr1, sc1), mask = _coords_to_mask(comp)
        templates[color] = mask
    assert templates
    return templates


def _find_walls(grid: np.ndarray, bg: int, panel: dict) -> list[dict]:
    panel_mask = np.zeros(grid.shape, dtype=bool)
    pr0, pc0, pr1, pc1 = panel["bbox"]
    panel_mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    walls = []
    wall_colors = set()
    for color in map(int, np.unique(grid)):
        if color == bg or color == panel["color"]:
            continue
        for coords in _components_mask((grid == color) & ~panel_mask):
            if len(coords) <= 1:
                continue
            info = _rect_frame_info(coords)
            if info is None or len(info["missing"]) != 1:
                continue
            side = _missing_side(info)
            r0, c0, r1, c1 = info["bbox"]
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            non_bg = np.argwhere(inner != bg)
            if non_bg.shape[0] != 1:
                continue
            sr, sc = map(int, non_bg[0])
            seed_color = int(inner[sr, sc])
            walls.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "wall_color": color,
                    "gap_side": side,
                    "seed_color": seed_color,
                    "seed_pos": (sr, sc),
                }
            )
            wall_colors.add(color)
    assert walls, "expected at least one wall"
    assert len(wall_colors) == 1, "all walls should share one structure color"
    return walls


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    panel = _find_display_panel(g, bg)
    templates = _extract_templates(g, panel, bg)
    walls = _find_walls(g, bg, panel)

    seed_colors = {wall["seed_color"] for wall in walls}
    assert seed_colors == set(templates), "seed/template colors must match one-to-one"

    out = g.copy()
    pr0, pc0, pr1, pc1 = panel["bbox"]
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = bg

    for wall in walls:
        r0, c0, r1, c1 = wall["bbox"]
        sprite = templates[wall["seed_color"]]
        rotated = np.rot90(sprite, -CW_TURNS[wall["gap_side"]])
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        assert rotated.shape[0] <= ih and rotated.shape[1] <= iw
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        stamp = patch[: rotated.shape[0], : rotated.shape[1]]
        stamp[rotated] = wall["seed_color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed color to its corner template, rotate the template by the wall gap direction, and stamp it inside the wall.

    Input structure:
    - bg = most common color.
    - One structure color draws several disconnected single-cell-thick wall enclosures, each with exactly one one-cell gap on its outer border and exactly one single-pixel seed of a unique color inside.
    - A different framed corner panel contains one monochrome template sprite for each seed color, separated by background.

    Rule:
    - Find the corner panel and read each template sprite keyed by its color.
    - For every wall enclosure, use the enclosed seed color to choose the matching template.
    - Rotate that template according to the wall gap side: top = 0 degrees, right = 90 degrees clockwise, bottom = 180 degrees, left = 270 degrees clockwise.
    - Place the rotated template into the enclosure interior with top-left alignment, keep the wall structure unchanged, and erase the whole corner panel to background in the output.

    Key insight: The seed color selects which template to use, while the single missing border cell independently encodes the template rotation.
    """
    return _solve(np.asarray(grid, dtype=int))


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    keys = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.add((rot.shape, rot.tobytes()))
    return keys


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    presets = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    for _ in range(120):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        total = h * w
        target = rng.randint(max(3, total // 2), total - 1)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        while len(cells) < target:
            frontier.clear()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        if len(coords) < 3:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if len(_rotation_keys(mask)) == 4:
            return mask
    return rng.choice(presets).copy()


def _pack_templates(items: list[tuple[int, np.ndarray]], rng: random.Random) -> tuple[tuple[int, int], dict[int, tuple[int, int]]]:
    rng.shuffle(items)
    n = len(items)
    cols = 1 if n == 1 or (n == 2 and rng.random() < 0.5) else 2
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, (_, mask) in enumerate(items):
        rr, cc = divmod(idx, cols)
        row_heights[rr] = max(row_heights[rr], mask.shape[0])
        col_widths[cc] = max(col_widths[cc], mask.shape[1])

    inner_h = sum(row_heights) + max(0, rows - 1) + 2
    inner_w = sum(col_widths) + max(0, cols - 1) + 2
    placements: dict[int, tuple[int, int]] = {}

    y = 1
    for rr in range(rows):
        x = 1
        for cc in range(cols):
            idx = rr * cols + cc
            if idx >= n:
                break
            color, mask = items[idx]
            dy = row_heights[rr] - mask.shape[0]
            dx = col_widths[cc] - mask.shape[1]
            placements[color] = (y + (rng.randint(0, dy) if dy else 0), x + (rng.randint(0, dx) if dx else 0))
            x += col_widths[cc] + 1
        y += row_heights[rr] + 1
    return (inner_h + 2, inner_w + 2), placements


def _choose_seed_position(
    interior_h: int,
    interior_w: int,
    rotated: np.ndarray,
    rng: random.Random,
) -> tuple[int, int]:
    inside = []
    outside = []
    for r in range(interior_h):
        for c in range(interior_w):
            covered = r < rotated.shape[0] and c < rotated.shape[1] and bool(rotated[r, c])
            (inside if covered else outside).append((r, c))
    use_inside = bool(inside) and (not outside or rng.random() < 0.5)
    return rng.choice(inside if use_inside else outside)


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    gh, gw = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > gh - 1 or c0 + w > gw - 1:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(gh, r0 + h + pad)
    cc1 = min(gw, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _draw_panel(
    grid: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
    panel_color: int,
    templates: dict[int, np.ndarray],
    placements: dict[int, tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = panel_bbox
    grid[r0, c0 : c1 + 1] = panel_color
    grid[r1, c0 : c1 + 1] = panel_color
    grid[r0 : r1 + 1, c0] = panel_color
    grid[r0 : r1 + 1, c1] = panel_color
    for color, mask in templates.items():
        sr, sc = placements[color]
        patch = grid[r0 + 1 + sr : r0 + 1 + sr + mask.shape[0], c0 + 1 + sc : c0 + 1 + sc + mask.shape[1]]
        patch[mask] = color


def _draw_wall(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    wall_color: int,
    gap_side: str,
    seed_color: int,
    seed_pos: tuple[int, int],
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color

    if gap_side == "top":
        gap = (r0, rng.randint(c0 + 1, c1 - 1))
    elif gap_side == "right":
        gap = (rng.randint(r0 + 1, r1 - 1), c1)
    elif gap_side == "bottom":
        gap = (r1, rng.randint(c0 + 1, c1 - 1))
    else:
        gap = (rng.randint(r0 + 1, r1 - 1), c0)
    grid[gap] = BG

    sr, sc = seed_pos
    grid[r0 + 1 + sr, c0 + 1 + sc] = seed_color


def _expected_output(
    grid: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
    regions: list[dict],
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = panel_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for region in regions:
        rr0, cc0, rr1, cc1 = region["bbox"]
        patch = out[rr0 + 1 : rr1, cc0 + 1 : cc1]
        mask = region["rotated"]
        stamp = patch[: mask.shape[0], : mask.shape[1]]
        stamp[mask] = region["seed_color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(2, 4)
    palette = list(range(1, 10))
    rng.shuffle(palette)
    wall_color = palette.pop()
    panel_color = palette.pop()
    seed_colors = [palette.pop() for _ in range(n_regions)]

    directions = rng.sample(list(SIDES), n_regions)
    templates: dict[int, np.ndarray] = {}
    regions = []
    for seed_color, gap_side in zip(seed_colors, directions):
        sprite = _random_sprite_mask(rng)
        rotated = np.rot90(sprite, -CW_TURNS[gap_side])
        interior_h = rng.randint(rotated.shape[0], min(6, rotated.shape[0] + 2))
        interior_w = rng.randint(rotated.shape[1], min(6, rotated.shape[1] + 2))
        seed_pos = _choose_seed_position(interior_h, interior_w, rotated, rng)
        templates[seed_color] = sprite
        regions.append(
            {
                "seed_color": seed_color,
                "gap_side": gap_side,
                "sprite": sprite,
                "rotated": rotated,
                "interior_h": interior_h,
                "interior_w": interior_w,
                "outer_h": interior_h + 2,
                "outer_w": interior_w + 2,
                "seed_pos": seed_pos,
            }
        )

    panel_size, template_positions = _pack_templates(list(templates.items()), rng)
    panel_h, panel_w = panel_size

    base_min = 18 + 2 * max(0, n_regions - 2)
    min_h = max(base_min, panel_h + 4)
    min_w = max(base_min, panel_w + 4)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        pr0, pc0 = 1, 1
    elif corner == "tr":
        pr0, pc0 = 1, grid_w - panel_w - 1
    elif corner == "bl":
        pr0, pc0 = grid_h - panel_h - 1, 1
    else:
        pr0, pc0 = grid_h - panel_h - 1, grid_w - panel_w - 1
    if pr0 < 1 or pc0 < 1:
        return None
    panel_bbox = (pr0, pc0, pr0 + panel_h - 1, pc0 + panel_w - 1)
    _draw_panel(grid, panel_bbox, panel_color, templates, template_positions)
    occupied[pr0 : pr0 + panel_h, pc0 : pc0 + panel_w] = True

    for region in sorted(regions, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        placed = False
        for _ in range(180):
            r0 = rng.randint(1, grid_h - region["outer_h"] - 1)
            c0 = rng.randint(1, grid_w - region["outer_w"] - 1)
            if not _can_place(occupied, r0, c0, region["outer_h"], region["outer_w"], pad=1):
                continue
            bbox = (r0, c0, r0 + region["outer_h"] - 1, c0 + region["outer_w"] - 1)
            _draw_wall(grid, bbox, wall_color, region["gap_side"], region["seed_color"], region["seed_pos"], rng)
            occupied[r0 : r0 + region["outer_h"], c0 : c0 + region["outer_w"]] = True
            region["bbox"] = bbox
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != BG:
        return None

    expected = _expected_output(grid, panel_bbox, regions)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
