from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
FRAME_COLORS = (2, 3, 7)
SPRITE_COLORS = (1, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
SPRITE_SIDE = 5
SPRITE_HOLE_COUNTS = (0, 1, 2)
FRAME_CHAMBERS = (1, 2, 3)
FRAME_SIDE_CHOICES = (11, 12, 13)
CANVAS_SIDE_RANGE = (26, 30)
PLACEMENT_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if coords.size:
        out[coords[:, 0], coords[:, 1]] = True
    return out


def _enclosed_background_components(solid_mask: np.ndarray) -> list[np.ndarray]:
    bg_parts = _components(~np.asarray(solid_mask, dtype=bool))
    return [part for part in bg_parts if not _touches_boundary(part, solid_mask.shape)]


def _count_enclosed_background(solid_mask: np.ndarray) -> int:
    return len(_enclosed_background_components(solid_mask))


def _classify_objects(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[dict] = []
    sprites: list[dict] = []
    for color in sorted(set(map(int, g.flatten())) - {bg}):
        box = _bbox(g == color)
        assert box is not None
        r0, c0, r1, c1 = box
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(np.sum(g == color))
        fill_ratio = area / float(h * w)
        info = {
            "color": int(color),
            "bbox": box,
            "patch": _crop_bbox(g, box),
            "h": h,
            "w": w,
            "fill_ratio": fill_ratio,
        }
        if h == w and h >= 8 and fill_ratio < 0.45:
            frames.append(info)
        else:
            sprites.append(info)
    return frames, sprites, bg


def _reconstruct_frame(local: np.ndarray, color: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = local.shape
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = True
    solid[h - 1, :] = True
    solid[:, 0] = True
    solid[:, w - 1] = True

    interior = np.zeros((h, w), dtype=bool)
    interior[1:h - 1, 1:w - 1] = local[1:h - 1, 1:w - 1] == int(color)
    divider_rows: set[int] = set()
    divider_cols: set[int] = set()
    for comp in _components(interior):
        rr = comp[:, 0]
        cc = comp[:, 1]
        row_span = int(rr.max() - rr.min())
        col_span = int(cc.max() - cc.min())
        if col_span >= row_span:
            divider_rows.add(int(np.median(rr)))
        else:
            divider_cols.add(int(np.median(cc)))

    for row in divider_rows:
        solid[int(row), 1:w - 1] = True
    for col in divider_cols:
        solid[1:h - 1, int(col)] = True

    chambers = _enclosed_background_components(solid)
    out = np.full((h, w), BG, dtype=int)
    out[solid] = int(color)
    return out, chambers


def _reconstruct_sprite(local: np.ndarray, color: int) -> tuple[np.ndarray, list[np.ndarray]]:
    observed = local == int(color)
    preserved_holes = _enclosed_background_components(observed)
    out = np.full(local.shape, int(color), dtype=int)
    for hole in preserved_holes:
        out[hole[:, 0], hole[:, 1]] = BG
    return out, preserved_holes


def _sorted_chambers(chambers: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(chambers, key=lambda coords: (int(coords[:, 1].min()), int(coords[:, 0].min())))


def _placement_in_chamber(chamber: np.ndarray, sprite_shape: tuple[int, int]) -> tuple[int, int] | None:
    sh, sw = map(int, sprite_shape)
    rr0 = int(chamber[:, 0].min())
    cc0 = int(chamber[:, 1].min())
    rr1 = int(chamber[:, 0].max())
    cc1 = int(chamber[:, 1].max())
    rel = np.array(chamber, dtype=int).copy()
    rel[:, 0] -= rr0
    rel[:, 1] -= cc0
    chamber_mask = _mask_from_coords(rel, (rr1 - rr0 + 1, cc1 - cc0 + 1))
    candidates: list[tuple[float, int, int]] = []
    target_cr = (rr0 + rr1) / 2.0
    target_cc = (cc0 + cc1) / 2.0
    for r in range(rr0, rr1 - sh + 2):
        for c in range(cc0, cc1 - sw + 2):
            local = chamber_mask[r - rr0 : r - rr0 + sh, c - cc0 : c - cc0 + sw]
            if local.shape != (sh, sw) or not bool(local.all()):
                continue
            sprite_cr = r + (sh - 1) / 2.0
            sprite_cc = c + (sw - 1) / 2.0
            score = abs(sprite_cr - target_cr) + abs(sprite_cc - target_cc)
            candidates.append((score, int(r), int(c)))
    if not candidates:
        return None
    _score, r_best, c_best = min(candidates)
    return int(r_best), int(c_best)


def _sprite_with_recolored_holes(sprite: np.ndarray, sprite_color: int, frame_color: int) -> np.ndarray:
    out = np.array(sprite, dtype=int).copy()
    out[out == BG] = int(frame_color)
    out[out == int(sprite_color)] = int(sprite_color)
    return out


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    frames_raw, sprites_raw, bg = _classify_objects(g)
    assert len(frames_raw) == 3
    assert len(sprites_raw) == 3

    frames: list[dict] = []
    for info in frames_raw:
        full, chambers = _reconstruct_frame(info["patch"], int(info["color"]))
        frames.append(
            {
                "color": int(info["color"]),
                "bbox": info["bbox"],
                "patch": full,
                "chambers": _sorted_chambers(chambers),
                "count": len(chambers),
            }
        )

    sprites: list[dict] = []
    for info in sprites_raw:
        full, holes = _reconstruct_sprite(info["patch"], int(info["color"]))
        sprites.append(
            {
                "color": int(info["color"]),
                "bbox": info["bbox"],
                "patch": full,
                "holes": holes,
                "count": len(holes),
            }
        )

    frame_by_count = {frame["count"] - 1: frame for frame in frames}
    sprite_by_count = {sprite["count"]: sprite for sprite in sprites}
    assert set(frame_by_count.keys()) == {0, 1, 2}
    assert set(sprite_by_count.keys()) == {0, 1, 2}

    out = np.full_like(g, int(bg))
    placements: dict[int, tuple[int, int]] = {}
    for key in (0, 1, 2):
        frame = frame_by_count[key]
        patch = np.asarray(frame["patch"], dtype=int)
        r0, c0, _r1, _c1 = frame["bbox"]
        _paste(out, patch, int(r0), int(c0), transparent=BG)

    for key in (0, 1, 2):
        frame = frame_by_count[key]
        sprite = sprite_by_count[key]
        placed = None
        for chamber in frame["chambers"]:
            top_left = _placement_in_chamber(chamber, sprite["patch"].shape)
            if top_left is not None:
                placed = top_left
                break
        assert placed is not None
        rr, cc = placed
        placements[int(sprite["color"])] = (int(rr), int(cc))
        recolored = _sprite_with_recolored_holes(
            np.asarray(sprite["patch"], dtype=int),
            int(sprite["color"]),
            int(frame["color"]),
        )
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        _paste(out, recolored, int(fr0) + int(rr), int(fc0) + int(cc), transparent=BG)

    return out, {"frames": frames, "sprites": sprites, "placements": placements}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse frames and sparse sprites, match them by post-completion hole count, and nest each sprite into its frame.

    Input structure:
    - The background is azure (8).
    - Three large same-color groups are sparse square frames: each color occupies the bbox border sparsely, and some frames also contain sparse same-color interior divider segments.
    - Three smaller same-color groups are sparse sprites inside tight bboxes, with their true holes indicated by enclosed background pockets in the sparse hint.

    Rule:
    - For each large group, complete its tight bbox border into a full square frame; each disconnected interior same-color segment extends to a full horizontal or vertical divider across the frame interior.
    - For each small group, fill its tight bbox completely with the sprite color except for background components already enclosed inside the sparse sprite hint; those enclosed components stay as holes.
    - Count sprite holes and frame chambers after completion, then pair hole count N with chamber count N+1.
    - Keep the completed frames in place, place each matched sprite centered in the first chamber that can contain it when chambers are ordered left-to-right then top-to-bottom, and recolor the sprite holes to the frame color.

    Key insight:
    - The sparse marks are not the final objects: bbox completion and divider completion change the topology, and the completed topology is the only reliable matching key.
    """
    out, _meta = _solve(np.asarray(grid, dtype=int))
    return out


def _sprite_full_pattern(hole_count: int) -> np.ndarray:
    sprite = np.full((SPRITE_SIDE, SPRITE_SIDE), 1, dtype=int)
    if hole_count == 0:
        return sprite
    interior_choices = [(1, 1), (1, 3), (3, 1), (3, 3)]
    if hole_count == 1:
        hole = random.choice(interior_choices + [(2, 2)])
        sprite[hole] = 0
        return sprite
    holes = random.choice([((1, 1), (3, 3)), ((1, 3), (3, 1)), ((1, 1), (3, 1)), ((1, 3), (3, 3))])
    for hole in holes:
        sprite[hole] = 0
    return sprite


def _sprite_support_mask(full_mask: np.ndarray) -> np.ndarray:
    support = np.zeros_like(full_mask, dtype=bool)
    h, w = full_mask.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for r, c in corners:
        support[r, c] = True
    holes = np.argwhere(~full_mask)
    for r, c in holes:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and full_mask[nr, nc]:
                support[nr, nc] = True
    return support


def _sprite_sparse_from_full(full_mask: np.ndarray, color: int) -> np.ndarray | None:
    fill_coords = [tuple(map(int, p)) for p in np.argwhere(full_mask)]
    fill_count = len(fill_coords)
    support = _sprite_support_mask(full_mask)
    support_coords = {tuple(map(int, p)) for p in np.argwhere(support)}
    min_keep = max(len(support_coords), int(np.ceil(0.5 * fill_count)))
    max_keep = max(min_keep, int(np.floor(0.7 * fill_count)))
    optional = [coord for coord in fill_coords if coord not in support_coords]
    intended_holes = _count_enclosed_background(full_mask)

    for _ in range(120):
        keep_n = random.randint(min_keep, max_keep)
        chosen = set(support_coords)
        extras_needed = keep_n - len(chosen)
        if extras_needed > len(optional):
            continue
        if extras_needed > 0:
            chosen.update(random.sample(optional, extras_needed))
        observed = np.zeros_like(full_mask, dtype=bool)
        for r, c in chosen:
            observed[int(r), int(c)] = True
        if _count_enclosed_background(observed) != intended_holes:
            continue
        if not observed[0, :].any() or not observed[-1, :].any() or not observed[:, 0].any() or not observed[:, -1].any():
            continue
        patch = np.full(full_mask.shape, BG, dtype=int)
        patch[observed] = int(color)
        return patch
    return None


def _frame_line_positions(side: int, chambers: int, sprite_side: int, orientation: str) -> list[int]:
    inner = side - 2
    if chambers == 1:
        return []
    if chambers == 2:
        first_len = random.randint(sprite_side, inner - 2)
        divider = 1 + first_len
        return [divider]
    assert chambers == 3
    first_len = random.randint(sprite_side, inner - 4)
    remaining = inner - 2 - first_len
    mid_len = random.randint(1, remaining - 1)
    divider1 = 1 + first_len
    divider2 = divider1 + 1 + mid_len
    return [divider1, divider2]


def _frame_full_patch(side: int, chambers: int, orientation: str) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    patch = np.full((side, side), BG, dtype=int)
    patch[0, :] = 1
    patch[side - 1, :] = 1
    patch[:, 0] = 1
    patch[:, side - 1] = 1
    lines: list[list[tuple[int, int]]] = []
    for pos in _frame_line_positions(side, chambers, SPRITE_SIDE, orientation):
        if orientation == "vertical":
            coords = [(r, int(pos)) for r in range(1, side - 1)]
        else:
            coords = [(int(pos), c) for c in range(1, side - 1)]
        lines.append(coords)
        for r, c in coords:
            patch[r, c] = 1
    return patch, lines


def _sample_border_coords(side: int, target_n: int) -> set[tuple[int, int]]:
    perimeter = []
    for c in range(side):
        perimeter.append((0, c))
        perimeter.append((side - 1, c))
    for r in range(1, side - 1):
        perimeter.append((r, 0))
        perimeter.append((r, side - 1))
    anchors = {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)}
    others = [cell for cell in perimeter if cell not in anchors]
    random.shuffle(others)
    chosen = set(anchors)
    chosen.update(others[: max(0, target_n - len(anchors))])
    return chosen


def _sample_line_segment(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    n = len(coords)
    seg_len = random.randint(max(2, int(np.ceil(0.4 * n))), max(2, int(np.floor(0.6 * n))))
    seg_len = min(seg_len, n - 2)
    start = random.randint(1, n - seg_len - 1)
    return set(coords[start : start + seg_len])


def _frame_sparse_from_full(side: int, chambers: int, color: int, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    full_mask, lines = _frame_full_patch(side, chambers, orientation)
    patch = np.full((side, side), BG, dtype=int)
    perimeter_count = 4 * side - 4
    border_target = random.randint(max(4, int(np.ceil(0.4 * perimeter_count))), int(np.floor(0.6 * perimeter_count)))
    for r, c in _sample_border_coords(side, border_target):
        patch[r, c] = int(color)
    for line in lines:
        for r, c in _sample_line_segment(line):
            patch[r, c] = int(color)
    full = np.full((side, side), BG, dtype=int)
    full[full_mask == 1] = int(color)
    return patch, full


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _random_place(canvas: np.ndarray, patch: np.ndarray, padding: int) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = canvas.shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if _can_place(canvas, patch, int(r0), int(c0), padding):
            return int(r0), int(c0)
    return None


def _expected_from_plan(shape: tuple[int, int], plan: dict) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for frame in plan["frames"]:
        fr, fc = frame["pos"]
        _paste(out, frame["full"], int(fr), int(fc), transparent=BG)

    for holes in (0, 1, 2):
        frame = next(item for item in plan["frames"] if item["chambers"] == holes + 1)
        sprite = next(item for item in plan["sprites"] if item["holes"] == holes)
        local_frame = np.asarray(frame["full"], dtype=int)
        solid = local_frame != BG
        chambers = _sorted_chambers(_enclosed_background_components(solid))
        chosen = None
        for chamber in chambers:
            top_left = _placement_in_chamber(chamber, sprite["full"].shape)
            if top_left is not None:
                chosen = top_left
                break
        assert chosen is not None
        rr, cc = chosen
        recolored = np.asarray(sprite["full"], dtype=int).copy()
        recolored[recolored == BG] = int(frame["color"])
        _paste(out, recolored, int(frame["pos"][0]) + int(rr), int(frame["pos"][1]) + int(cc), transparent=BG)
    return out


def _build_plan() -> dict | None:
    H = random.randint(*CANVAS_SIDE_RANGE)
    W = random.randint(*CANVAS_SIDE_RANGE)
    canvas = np.full((H, W), BG, dtype=int)

    sprite_colors = random.sample(list(SPRITE_COLORS), 3)
    frame_colors = random.sample(list(FRAME_COLORS), 3)

    sprites = []
    for holes, color in zip(SPRITE_HOLE_COUNTS, sprite_colors):
        full01 = _sprite_full_pattern(int(holes))
        sparse = _sprite_sparse_from_full(full01, int(color))
        if sparse is None:
            return None
        full = np.full(full01.shape, BG, dtype=int)
        full[full01 == 1] = int(color)
        sprites.append({"holes": int(holes), "color": int(color), "sparse": sparse, "full": full, "mask01": full01})

    frames = []
    for chambers, color in zip(FRAME_CHAMBERS, frame_colors):
        side = random.choice(FRAME_SIDE_CHOICES)
        orientation = random.choice(["vertical", "horizontal"]) if chambers > 1 else "vertical"
        sparse, full = _frame_sparse_from_full(int(side), int(chambers), int(color), orientation)
        frames.append(
            {
                "chambers": int(chambers),
                "color": int(color),
                "side": int(side),
                "orientation": orientation,
                "sparse": sparse,
                "full": full,
            }
        )

    items = [("frame", item) for item in frames] + [("sprite", item) for item in sprites]
    random.shuffle(items)
    for kind, item in items:
        patch = item["sparse"]
        pos = _random_place(canvas, patch, PLACEMENT_PADDING)
        if pos is None:
            return None
        item["pos"] = pos
        _paste(canvas, patch, int(pos[0]), int(pos[1]), transparent=BG)

    return {"input": canvas, "frames": frames, "sprites": sprites}


def _validate_plan(plan: dict) -> bool:
    if {item["holes"] for item in plan["sprites"]} != {0, 1, 2}:
        return False
    if {item["chambers"] for item in plan["frames"]} != {1, 2, 3}:
        return False
    for sprite in plan["sprites"]:
        if _count_enclosed_background(sprite["full"] != BG) != int(sprite["holes"]):
            return False
        if _count_enclosed_background(sprite["sparse"] == int(sprite["color"])) != int(sprite["holes"]):
            return False
    topology_changed = 0
    for frame in plan["frames"]:
        full_count = _count_enclosed_background(frame["full"] != BG)
        raw_count = _count_enclosed_background(frame["sparse"] == int(frame["color"]))
        if full_count != int(frame["chambers"]):
            return False
        if int(frame["chambers"]) > 1 and raw_count == full_count:
            return False
        if raw_count != full_count:
            topology_changed += 1
    if topology_changed < 2:
        return False

    expected = _expected_from_plan(plan["input"].shape, plan)
    solved = main(plan["input"])
    if not np.array_equal(expected, solved):
        return False

    frames_raw, sprites_raw, _bg = _classify_objects(plan["input"])
    if len(frames_raw) != 3 or len(sprites_raw) != 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        plan = _build_plan()
        if plan is None:
            continue
        if _validate_plan(plan):
            return np.asarray(plan["input"], dtype=int)
    raise RuntimeError("failed to generate a valid sparse-hole matching instance")
