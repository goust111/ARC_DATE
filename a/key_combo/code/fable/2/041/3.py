from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def find_components(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    comps: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in _mask_components(g == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[rr + int(r0), cc + int(c0)] = int(color)


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_single_component(mask: np.ndarray) -> bool:
    parts = _mask_components(mask)
    return len(parts) == 1


def _random_dual_mask(height: int, width: int) -> np.ndarray:
    for _ in range(800):
        p = random.uniform(0.35, 0.65)
        notch_mask = np.random.random((height, width)) < p
        if int(notch_mask.sum()) < 2 or int((~notch_mask).sum()) < 2:
            continue
        if not _is_single_component(notch_mask):
            continue
        if not _is_single_component(~notch_mask):
            continue
        if not _touches_all_edges(notch_mask) or not _touches_all_edges(~notch_mask):
            continue
        return notch_mask
    raise RuntimeError("failed to sample a dual-connected notch mask")


def _random_other_mask(height: int, width: int, forbidden: set[bytes]) -> np.ndarray:
    for _ in range(800):
        p = random.uniform(0.30, 0.70)
        sprite_mask = np.random.random((height, width)) < p
        if sprite_mask.tobytes() in forbidden:
            continue
        if int(sprite_mask.sum()) < 2 or int((~sprite_mask).sum()) < 2:
            continue
        if not _touches_all_edges(sprite_mask):
            continue
        if not _is_single_component(sprite_mask):
            continue
        return sprite_mask
    raise RuntimeError("failed to sample a jam mask")


def _wall_side(box: BBox, width: int) -> str:
    touch_left = int(box[1]) == 0
    touch_right = int(box[3]) == width - 1
    assert touch_left ^ touch_right, "wall must touch exactly one vertical border"
    return "left" if touch_left else "right"


def _build_wall_mask(shape: tuple[int, int], wall_coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[wall_coords[:, 0], wall_coords[:, 1]] = True
    return mask


def _extract_notches(grid: np.ndarray, wall_mask: np.ndarray, wall_box: BBox, side: str, bg: int) -> list[dict]:
    r0, c0, r1, c1 = map(int, wall_box)
    inside = np.zeros(grid.shape, dtype=bool)
    inside[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_inside = inside & (np.asarray(grid) == int(bg)) & (~wall_mask)
    edge_col = c1 if side == "left" else c0
    notches: list[dict] = []
    for coords in _mask_components(bg_inside):
        if not np.any(coords[:, 1] == edge_col):
            continue
        box = bbox_coords(coords)
        mask = _coords_to_mask(coords)
        notches.append({"coords": coords, "bbox": box, "mask": mask})
    notches.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return notches


def _slide_coords(coords: np.ndarray, wall_mask: np.ndarray, dc: int, shape: tuple[int, int]) -> np.ndarray:
    moved = coords.copy()
    _height, width = map(int, shape)
    while True:
        trial = moved.copy()
        trial[:, 1] += int(dc)
        if int(trial[:, 1].min()) < 0 or int(trial[:, 1].max()) >= width:
            return moved
        if wall_mask[trial[:, 0], trial[:, 1]].any():
            return moved
        moved = trial


def _pair_notch(sprite_box: BBox, notches: list[dict]) -> dict:
    sr0, _sc0, sr1, _sc1 = map(int, sprite_box)
    hits = [
        notch
        for notch in notches
        if not (sr1 < int(notch["bbox"][0]) or int(notch["bbox"][2]) < sr0)
    ]
    assert len(hits) == 1, "each sprite must face exactly one notch"
    return hits[0]


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg)
    assert comps, "expected at least one non-background component"

    edge_comps = [comp for comp in comps if int(comp["bbox"][1]) == 0 or int(comp["bbox"][3]) == g.shape[1] - 1]
    assert edge_comps, "expected a wall touching a side border"
    wall = max(edge_comps, key=lambda comp: (int(comp["area"]), int(comp["bbox"][2]) - int(comp["bbox"][0])))
    side = _wall_side(wall["bbox"], g.shape[1])
    wall_mask = _build_wall_mask(g.shape, wall["coords"])
    notches = _extract_notches(g, wall_mask, wall["bbox"], side, bg)
    assert notches, "expected at least one notch carved into the wall"

    sprites = [comp for comp in comps if comp is not wall]
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    return {
        "grid": g,
        "bg": int(bg),
        "wall": wall,
        "wall_color": int(wall["color"]),
        "side": side,
        "wall_mask": wall_mask,
        "notches": notches,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite horizontally toward the side wall and absorb only the ones that are notch complements.

    Input structure:
    - One side of the grid is occupied by a thick monochrome wall.
    - The wall's inner vertical edge contains one or more background-colored notches.
    - Outside the wall sit several monochrome sprites, each sharing the row interval of exactly one notch.

    Rule:
    - Slide every sprite horizontally toward the wall until the next step would hit wall material.
    - Translate the stopped sprite straight onto the facing notch bbox.
    - If the sprite is the exact negative-shape complement of the notch, erase the sprite and fill that notch bbox with the wall color; otherwise leave the sprite at its collision position in its original color.

    Key insight:
    - The fitting sprite is not the same shape as the visible cavity: when translated onto the notch bbox, it must occupy exactly the cells the notch does not.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = int(scene["bg"])
    wall_color = int(scene["wall_color"])
    dc = 1 if scene["side"] == "right" else -1

    out = g.copy()
    for sprite in scene["sprites"]:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg

    for sprite in scene["sprites"]:
        notch = _pair_notch(sprite["bbox"], scene["notches"])
        moved = _slide_coords(sprite["coords"], scene["wall_mask"], dc, g.shape)
        sprite_mask = _coords_to_mask(moved)
        fits = sprite_mask.shape == notch["mask"].shape and np.array_equal(sprite_mask, ~notch["mask"])
        if fits:
            r0, c0, r1, c1 = map(int, notch["bbox"])
            out[r0 : r1 + 1, c0 : c1 + 1] = wall_color
        else:
            out[moved[:, 0], moved[:, 1]] = int(sprite["color"])

    return out


GEN_RETRIES = 400
NOTCH_COUNT_RANGE = (2, 4)
NOTCH_H_RANGE = (3, 5)
NOTCH_W_RANGE = (3, 5)
GRID_H_RANGE = (14, 28)
WALL_EXTRA_RANGE = (2, 4)
OUTER_EXTRA_RANGE = (6, 12)
MIN_ROW_GAP = 1


def _row_layout(count: int) -> tuple[int, list[tuple[int, int]]]:
    for _ in range(400):
        heights = [random.randint(*NOTCH_H_RANGE) for _ in range(count)]
        min_total = sum(heights) + (count + 1) * MIN_ROW_GAP
        if min_total > GRID_H_RANGE[1]:
            continue
        height = random.randint(max(GRID_H_RANGE[0], min_total), GRID_H_RANGE[1])
        slack = height - min_total
        gaps = [MIN_ROW_GAP] * (count + 1)
        for _ in range(slack):
            gaps[random.randrange(count + 1)] += 1
        rows: list[tuple[int, int]] = []
        cursor = gaps[0]
        for idx, block_h in enumerate(heights):
            rows.append((cursor, cursor + block_h - 1))
            cursor += block_h + gaps[idx + 1]
        return height, rows
    raise RuntimeError("failed to place notch row bands")


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full((spec["height"], spec["width"]), spec["bg"], dtype=int)
    wall_cols = slice(0, spec["wall_thickness"]) if spec["side"] == "left" else slice(spec["width"] - spec["wall_thickness"], spec["width"])
    grid[:, wall_cols] = spec["wall_color"]

    for item in spec["items"]:
        r0, c0, r1, c1 = item["notch_bbox"]
        notch_patch = np.full((r1 - r0 + 1, c1 - c0 + 1), spec["wall_color"], dtype=int)
        notch_patch[item["notch_mask"]] = spec["bg"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = notch_patch
        _paint_mask(grid, item["sprite_mask"], r0, item["sprite_c0"], item["sprite_color"])

    return grid


def _expected_output(spec: dict, grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    wall_mask = np.asarray(grid) == int(spec["wall_color"])
    for item in spec["items"]:
        r0, c0, r1, c1 = item["notch_bbox"]
        sprite_coords = np.argwhere(item["sprite_mask"]) + np.array([r0, item["sprite_c0"]], dtype=int)
        out[sprite_coords[:, 0], sprite_coords[:, 1]] = int(spec["bg"])
        moved = _slide_coords(sprite_coords, wall_mask, item["dc"], grid.shape)
        if item["fits"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = spec["wall_color"]
        else:
            out[moved[:, 0], moved[:, 1]] = item["sprite_color"]
    return out


def _semantic_flags(scene: dict) -> list[bool]:
    flags: list[bool] = []
    for sprite in scene["sprites"]:
        notch = _pair_notch(sprite["bbox"], scene["notches"])
        sprite_mask = _coords_to_mask(sprite["coords"])
        flags.append(
            sprite_mask.shape == notch["mask"].shape
            and np.array_equal(sprite_mask, ~notch["mask"])
        )
    return flags


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(*NOTCH_COUNT_RANGE)
    height, rows = _row_layout(count)
    widths = [random.randint(*NOTCH_W_RANGE) for _ in range(count)]
    wall_thickness = max(widths) + random.randint(*WALL_EXTRA_RANGE)
    width = wall_thickness + max(widths) + random.randint(*OUTER_EXTRA_RANGE)
    if width > 30 or wall_thickness >= width:
        return None

    side = random.choice(("left", "right"))
    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    wall_color = random.choice(palette)
    palette.remove(wall_color)
    if len(palette) < count:
        return None
    sprite_colors = random.sample(palette, count)

    fit_flags = [True, False] + [random.random() < 0.5 for _ in range(count - 2)]
    random.shuffle(fit_flags)
    items: list[dict] = []

    for idx, ((r0, r1), notch_w, fits) in enumerate(zip(rows, widths, fit_flags)):
        h = r1 - r0 + 1
        base_mask = _random_dual_mask(h, notch_w)
        notch_mask = np.fliplr(base_mask) if side == "left" else base_mask
        if fits:
            sprite_mask = ~notch_mask
        elif random.random() < 0.55:
            sprite_mask = notch_mask.copy()
        else:
            sprite_mask = _random_other_mask(
                h,
                notch_w,
                {notch_mask.tobytes(), (~notch_mask).tobytes()},
            )
        sprite_w = sprite_mask.shape[1]

        if side == "left":
            notch_bbox = (r0, wall_thickness - notch_w, r1, wall_thickness - 1)
            sprite_lo = wall_thickness + 2
            sprite_hi = width - sprite_w - 1
            dc = -1
        else:
            notch_bbox = (r0, width - wall_thickness, r1, width - wall_thickness + notch_w - 1)
            sprite_lo = 1
            sprite_hi = width - wall_thickness - sprite_w - 2
            dc = 1
        if sprite_lo > sprite_hi:
            return None
        sprite_c0 = random.randint(sprite_lo, sprite_hi)
        items.append(
            {
                "notch_bbox": notch_bbox,
                "notch_mask": notch_mask,
                "sprite_mask": sprite_mask,
                "sprite_c0": sprite_c0,
                "sprite_color": sprite_colors[idx],
                "fits": fits,
                "dc": dc,
            }
        )

    spec = {
        "height": height,
        "width": width,
        "bg": bg,
        "wall_color": wall_color,
        "side": side,
        "wall_thickness": wall_thickness,
        "items": items,
    }
    grid = _render_input(spec)
    expected = _expected_output(spec, grid)

    try:
        scene = _analyze_scene(grid)
    except AssertionError:
        return None
    if len(scene["sprites"]) != count or len(scene["notches"]) != count:
        return None
    parsed_flags = _semantic_flags(scene)
    if parsed_flags.count(True) < 1 or parsed_flags.count(False) < 1:
        return None
    if parsed_flags != fit_flags:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-and-notch puzzle")
