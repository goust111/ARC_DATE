from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
SPRITE_COLORS = (1, 2, 3, 4)
OPEN_SIDES = ("top", "bottom", "left", "right")
MAX_GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _find_components(
    grid: np.ndarray,
    *,
    ignored_colors: set[int],
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignored_colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _indicator_axis(grid: np.ndarray) -> str:
    frames = [comp for comp in _find_components(grid, ignored_colors={BG}) if comp["color"] == FRAME]
    assert len(frames) == 1, "expected exactly one gray frame indicator"
    crop = _crop_bbox(grid, frames[0]["bbox"])

    missing = []
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        raise AssertionError("indicator crop is too small")

    if not np.any(crop[0, 1:-1] == FRAME):
        missing.append("top")
    if not np.any(crop[-1, 1:-1] == FRAME):
        missing.append("bottom")
    if not np.any(crop[1:-1, 0] == FRAME):
        missing.append("left")
    if not np.any(crop[1:-1, -1] == FRAME):
        missing.append("right")

    assert len(missing) == 1, "indicator must have exactly one missing side"
    return "vertical" if missing[0] in {"top", "bottom"} else "horizontal"


def _pack_crops(crops: list[np.ndarray], axis: str) -> np.ndarray:
    assert crops
    if axis == "horizontal":
        out_h = max(int(c.shape[0]) for c in crops)
        out_w = sum(int(c.shape[1]) for c in crops)
        out = np.zeros((out_h, out_w), dtype=int)
        cursor = 0
        for crop in crops:
            h, w = crop.shape
            _paste(out, crop, 0, cursor)
            cursor += w
        return out

    out_h = sum(int(c.shape[0]) for c in crops)
    out_w = max(int(c.shape[1]) for c in crops)
    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for crop in crops:
        h, w = crop.shape
        _paste(out, crop, cursor, 0)
        cursor += h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the gray frame opening to choose a packing axis, then crop, area-sort, and tightly pack the colored sprites.

    Input structure:
    - A black background contains one gray outline frame near the top-left; exactly one of its four sides is fully missing.
    - The rest of the grid contains 3-4 separated colored 4-connected sprites, each in a distinct non-gray color.
    - The gray frame is only an indicator and is not part of the output.

    Rule:
    - Detect the missing side of the gray frame: a top/bottom opening means vertical packing, and a left/right opening means horizontal packing.
    - For every non-gray sprite other than the indicator, crop its minimal bounding box.
    - Sort those crops by sprite area (number of non-black cells) from smallest to largest.
    - Pack the sorted crops flush with no gaps along the indicated axis: left-to-right for horizontal, top-to-bottom for vertical.

    Key insight:
    - The frame opening encodes the packing axis, while the sprites themselves must be cropped and ordered by area before concatenation.
    """
    g = np.asarray(grid, dtype=int)
    axis = _indicator_axis(g)

    sprites = [
        comp
        for comp in _find_components(g, ignored_colors={BG, FRAME})
        if comp["color"] in SPRITE_COLORS
    ]
    assert 3 <= len(sprites) <= 4, "expected 3-4 colored sprites"

    ordered_crops = [
        _crop_bbox(g, comp["bbox"])
        for comp in sorted(sprites, key=lambda comp: (comp["area"], comp["color"]))
    ]
    return _pack_crops(ordered_crops, axis)


def _shape_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _shape_with_exact_bbox(h: int, w: int, area: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(300):
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            weights = []
            for r, c in frontier:
                score = 1
                if r in (0, h - 1):
                    score += 2
                if c in (0, w - 1):
                    score += 2
                weights.append(score)
            cells.add(frontier[rng.choices(range(len(frontier)), weights=weights, k=1)[0]])
        if len(cells) != area:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue
        mask = np.zeros((h, w), dtype=int)
        for r, c in cells:
            mask[r, c] = 1
        return mask
    return None


def _make_sprite_specs(rng: random.Random, count: int) -> list[dict] | None:
    area_choices = sorted(rng.sample(list(range(4, 10)), count))
    specs = []
    for area in area_choices:
        made = None
        dims = [
            (h, w)
            for h in range(2, 6)
            for w in range(2, 6)
            if h * w >= area and h * w <= 12
        ]
        rng.shuffle(dims)
        for h, w in dims:
            mask = _shape_with_exact_bbox(h, w, area, rng)
            if mask is not None:
                made = {
                    "area": area,
                    "bbox_shape": (h, w),
                    "bbox_area": h * w,
                    "mask": mask,
                }
                break
        if made is None:
            return None
        specs.append(made)

    heights = {spec["bbox_shape"][0] for spec in specs}
    widths = {spec["bbox_shape"][1] for spec in specs}
    if len(heights) < 2 or len(widths) < 2:
        return None

    area_order = list(range(count))
    bbox_order = sorted(range(count), key=lambda i: (specs[i]["bbox_area"], specs[i]["bbox_shape"]))
    if area_order == bbox_order:
        return None
    return specs


def _colorize_specs(specs: list[dict], rng: random.Random) -> list[dict]:
    colors = list(SPRITE_COLORS)
    rng.shuffle(colors)
    out = []
    for spec, color in zip(specs, colors):
        crop = spec["mask"] * int(color)
        out.append(
            {
                "area": int(spec["area"]),
                "bbox_area": int(spec["bbox_area"]),
                "crop": crop.astype(int),
                "color": int(color),
            }
        )
    return out


def _frame_patch(open_side: str) -> np.ndarray:
    patch = np.zeros((4, 4), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    if open_side == "top":
        patch[0, 1:-1] = BG
    elif open_side == "bottom":
        patch[-1, 1:-1] = BG
    elif open_side == "left":
        patch[1:-1, 0] = BG
    else:
        patch[1:-1, -1] = BG
    return patch


def _occupied_with_halo(
    crop: np.ndarray,
    r0: int,
    c0: int,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = set()
    halo = set()
    for r, c in np.argwhere(crop != BG):
        rr = int(r0 + r)
        cc = int(c0 + c)
        cells.add((rr, cc))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if _in_bounds(shape, nr, nc):
                    halo.add((nr, nc))
    return cells, halo


def _place_frame(grid: np.ndarray, patch: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    h, w = grid.shape
    positions = [(r, c) for r in range(0, min(3, h - 3)) for c in range(0, min(3, w - 3))]
    rng.shuffle(positions)
    r0, c0 = positions[0]
    _paste(grid, patch, r0, c0)
    return r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1


def _placement_candidates(
    grid_shape: tuple[int, int],
    crop: np.ndarray,
    axis: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    gh, gw = grid_shape
    h, w = crop.shape
    rows = list(range(gh - h + 1))
    cols = list(range(gw - w + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)

    pairs = []
    if axis == "horizontal":
        for r in rows:
            for c in cols:
                pairs.append((r, c))
    else:
        for c in cols:
            for r in rows:
                pairs.append((r, c))
    rng.shuffle(pairs)
    return pairs


def _place_sprites(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    sprites: list[dict],
    axis: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    fh0, fw0, fh1, fw1 = frame_bbox
    blocked = set()
    for r in range(max(0, fh0 - 1), min(grid_shape[0], fh1 + 2)):
        for c in range(max(0, fw0 - 1), min(grid_shape[1], fw1 + 2)):
            blocked.add((r, c))

    for _ in range(250):
        positions: list[tuple[int, int] | None] = [None] * len(sprites)
        occupied_halo = set(blocked)
        order = list(range(len(sprites)))
        rng.shuffle(order)

        ok = True
        for idx in order:
            crop = sprites[idx]["crop"]
            found = None
            for r0, c0 in _placement_candidates(grid_shape, crop, axis, rng):
                cells, halo = _occupied_with_halo(crop, r0, c0, grid_shape)
                if cells & occupied_halo:
                    continue
                found = (r0, c0, halo)
                break
            if found is None:
                ok = False
                break
            positions[idx] = (found[0], found[1])
            occupied_halo |= found[2]

        if not ok:
            continue

        placed = [pos for pos in positions if pos is not None]
        if axis == "horizontal":
            xs = [c for _, c in placed]
            if len(set(xs)) != len(xs):
                continue
        else:
            ys = [r for r, _ in placed]
            if len(set(ys)) != len(ys):
                continue

        centers = []
        for sprite, (r0, c0) in zip(sprites, placed):
            h, w = sprite["crop"].shape
            centers.append((r0 + (h - 1) / 2, c0 + (w - 1) / 2))
        row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
        col_span = max(c for _, c in centers) - min(c for _, c in centers)
        if axis == "horizontal":
            if not (row_span > col_span):
                continue
        else:
            if not (col_span > row_span):
                continue

        coord_order = (
            sorted(range(len(placed)), key=lambda i: (placed[i][1], placed[i][0]))
            if axis == "horizontal"
            else sorted(range(len(placed)), key=lambda i: (placed[i][0], placed[i][1]))
        )
        if coord_order == list(range(len(placed))):
            continue
        return [(int(r), int(c)) for r, c in placed]
    return None


def _render_input(
    grid_shape: tuple[int, int],
    open_side: str,
    sprites: list[dict],
    positions: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    frame_bbox = _place_frame(grid, _frame_patch(open_side), rng)
    for sprite, (r0, c0) in zip(sprites, positions):
        _paste(grid, sprite["crop"], r0, c0)
    return grid


def _expected_output(sprites: list[dict], axis: str) -> np.ndarray:
    ordered = [sprite["crop"] for sprite in sorted(sprites, key=lambda s: (s["area"], s["color"]))]
    return _pack_crops(ordered, axis)


def _alternative_position_output(
    sprites: list[dict],
    positions: list[tuple[int, int]],
    axis: str,
) -> np.ndarray:
    order = (
        sorted(range(len(sprites)), key=lambda i: (positions[i][1], positions[i][0]))
        if axis == "horizontal"
        else sorted(range(len(sprites)), key=lambda i: (positions[i][0], positions[i][1]))
    )
    return _pack_crops([sprites[i]["crop"] for i in order], axis)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    grid_shape = (rng.randint(13, 18), rng.randint(13, 18))
    open_side = rng.choice(OPEN_SIDES)
    axis = "vertical" if open_side in {"top", "bottom"} else "horizontal"
    count = rng.randint(3, 4)

    specs = _make_sprite_specs(rng, count)
    if specs is None:
        return None
    sprites = _colorize_specs(specs, rng)

    temp = np.zeros(grid_shape, dtype=int)
    frame_bbox = _place_frame(temp, _frame_patch(open_side), rng)
    positions = _place_sprites(grid_shape, frame_bbox, sprites, axis, rng)
    if positions is None:
        return None

    grid = np.zeros(grid_shape, dtype=int)
    _paste(grid, _frame_patch(open_side), frame_bbox[0], frame_bbox[1])
    for sprite, (r0, c0) in zip(sprites, positions):
        _paste(grid, sprite["crop"], r0, c0)

    out = _expected_output(sprites, axis)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    if np.array_equal(out, _pack_crops([s["crop"] for s in sorted(sprites, key=lambda s: (s["area"], s["color"]))], "horizontal" if axis == "vertical" else "vertical")):
        return None
    if np.array_equal(out, _alternative_position_output(sprites, positions, axis)):
        return None

    bbox_order = sorted(range(len(sprites)), key=lambda i: (sprites[i]["bbox_area"], sprites[i]["area"]))
    if np.array_equal(out, _pack_crops([sprites[i]["crop"] for i in bbox_order], axis)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate input for puzzle 9")
