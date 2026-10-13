from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 6
ORANGE = 7
TEAL = 8
SPRITE_COLORS = (1, 2, 3, 4, 5, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
MARKER_BOX = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _rect_conflicts(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _shape_mask(h: int, w: int, area: int) -> np.ndarray:
    min_area = h + w - 1
    max_area = min(12, h * w)
    assert min_area <= area <= max_area

    mask = np.zeros((h, w), dtype=bool)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[0, :] = True
        mask[:, 0] = True
    elif corner == "tr":
        mask[0, :] = True
        mask[:, w - 1] = True
    elif corner == "bl":
        mask[h - 1, :] = True
        mask[:, 0] = True
    else:
        mask[h - 1, :] = True
        mask[:, w - 1] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    coords = np.argwhere(mask)
    assert _bbox(coords) == (0, 0, h - 1, w - 1)
    return mask


def _sprite_patch(h: int, w: int, area: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[_shape_mask(h, w, area)] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two marker colors, filter outside sprites by area, and embed only the size-matching survivors into the template cavities.

    Input structure:
    - A black background contains one large magenta template frame with several rectangular black cavities inside its bbox.
    - Outside the frame are several uniquely colored single-color 4-connected sprites, plus orange and teal marker pixels whose positions do not touch the frame logic.
    - Sprite bounding boxes are the units compared against cavity sizes, while sprite filled areas are used only for filtering.

    Rule:
    - Count orange pixels and teal pixels, then let lo/hi be the smaller/larger of those two counts.
    - Keep only outside sprites whose filled area lies in the inclusive interval [lo, hi].
    - Match each kept sprite to one cavity whose height/width equals the sprite bbox, allowing a 90-degree rotation if needed, then paste the sprite into that cavity with top-left alignment.
    - Leave unmatched cavities black and return only the tight bbox crop of the magenta template frame.

    Key insight:
    - The marker pixels contribute only numeric thresholds; the visible placement step is controlled by a second criterion, namely bbox-size matching between the filtered sprites and the internal cavities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orange_count = int(np.count_nonzero(g == ORANGE))
    teal_count = int(np.count_nonzero(g == TEAL))
    lo = min(orange_count, teal_count)
    hi = max(orange_count, teal_count)
    assert 3 <= lo <= hi

    frame_components = _components(g == FRAME)
    assert frame_components
    frame_coords = max(frame_components, key=lambda coords: int(coords.shape[0]))
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    out = np.where(crop == FRAME, FRAME, BG).astype(int)

    cavities: list[dict[str, int]] = []
    for coords in _components(crop == BG):
        if _touches_border(coords, crop.shape):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        cavity = crop[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(cavity == BG)
        cavities.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
        )
    cavities.sort(key=lambda item: (item["r0"], item["c0"]))

    sprites: list[dict[str, object]] = []
    for color in np.unique(g):
        col = int(color)
        if col in (BG, FRAME, ORANGE, TEAL):
            continue
        for coords in _components(g == col):
            r0, c0, r1, c1 = _bbox(coords)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            sprites.append(
                {
                    "color": col,
                    "area": int(coords.shape[0]),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "patch": patch,
                }
            )

    kept = [sprite for sprite in sprites if lo <= int(sprite["area"]) <= hi]
    candidates: list[list[tuple[int, bool]]] = []
    for sprite in kept:
        options: list[tuple[int, bool]] = []
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        for cavity_idx, cavity in enumerate(cavities):
            if (sh, sw) == (int(cavity["h"]), int(cavity["w"])):
                options.append((cavity_idx, False))
            elif (sh, sw) == (int(cavity["w"]), int(cavity["h"])):
                options.append((cavity_idx, True))
        candidates.append(options)

    owners: list[tuple[int, bool] | None] = [None] * len(cavities)

    def augment(sprite_idx: int, seen: list[bool]) -> bool:
        for cavity_idx, rotate in candidates[sprite_idx]:
            if seen[cavity_idx]:
                continue
            seen[cavity_idx] = True
            owner = owners[cavity_idx]
            if owner is None or augment(owner[0], seen):
                owners[cavity_idx] = (sprite_idx, rotate)
                return True
        return False

    order = sorted(range(len(kept)), key=lambda idx: (len(candidates[idx]), -int(kept[idx]["area"])))
    for sprite_idx in order:
        augment(sprite_idx, [False] * len(cavities))

    for cavity_idx, assignment in enumerate(owners):
        if assignment is None:
            continue
        sprite_idx, rotate = assignment
        cavity = cavities[cavity_idx]
        patch = np.array(kept[sprite_idx]["patch"], dtype=int)
        if rotate:
            patch = np.rot90(patch, 1)
        _place_patch(out, patch, int(cavity["r0"]), int(cavity["c0"]))

    return out


INTERVAL_CHOICES = ((3, 6), (3, 7), (4, 6), (4, 7), (4, 8), (5, 8))


def _dim_infos(lo: int, hi: int) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for h in range(2, 5):
        for w in range(h, 6):
            min_area = h + w - 1
            max_area = min(12, h * w)
            areas = list(range(min_area, max_area + 1))
            pass_areas = [area for area in areas if lo <= area <= hi]
            out_areas = [area for area in areas if area < lo or area > hi]
            if pass_areas:
                infos.append(
                    {
                        "h": h,
                        "w": w,
                        "pass_areas": pass_areas,
                        "out_areas": out_areas,
                        "key": (h, w),
                    }
                )
    return infos


def _build_frame(cavities: list[dict[str, int]]) -> tuple[np.ndarray, list[dict[str, int]]]:
    rows = (len(cavities) + 1) // 2
    row_groups = [cavities[2 * row : 2 * row + 2] for row in range(rows)]
    row_heights = [max(item["h"] for item in group) for group in row_groups]

    col_widths: list[int] = []
    for col in range(2):
        widths = [group[col]["w"] for group in row_groups if col < len(group)]
        col_widths.append(max(widths) if widths else 0)

    frame_h = sum(row_heights) + rows + 1
    frame_w = sum(col_widths) + 3
    if frame_h < 10 or frame_w < 10 or frame_h > 16 or frame_w > 13:
        raise ValueError("frame layout exceeds the intended template size")

    frame = np.full((frame_h, frame_w), FRAME, dtype=int)
    placed: list[dict[str, int]] = []
    for idx, cavity in enumerate(cavities):
        row = idx // 2
        col = idx % 2
        slot_r0 = 1 + sum(row_heights[:row]) + row
        slot_c0 = 1 + sum(col_widths[:col]) + col
        slack_r = row_heights[row] - cavity["h"]
        slack_c = col_widths[col] - cavity["w"]
        r0 = slot_r0 + random.randint(0, slack_r)
        c0 = slot_c0 + random.randint(0, slack_c)
        r1 = r0 + cavity["h"] - 1
        c1 = c0 + cavity["w"] - 1
        frame[r0 : r1 + 1, c0 : c1 + 1] = BG
        item = dict(cavity)
        item.update({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
        placed.append(item)
    return frame, placed


def _choose_layout() -> tuple[np.ndarray, list[dict[str, int]], list[dict[str, object]], int, int] | None:
    lo, hi = random.choice(INTERVAL_CHOICES)
    infos = _dim_infos(lo, hi)
    n_cavities = random.choice((4, 5))
    for _ in range(120):
        selected = random.sample(infos, n_cavities)
        lo_idxs = [idx for idx, info in enumerate(selected) if lo in info["pass_areas"]]
        hi_idxs = [idx for idx, info in enumerate(selected) if hi in info["pass_areas"]]
        twin_idxs = [idx for idx, info in enumerate(selected) if info["out_areas"]]
        rotate_idxs = [idx for idx, info in enumerate(selected) if int(info["h"]) != int(info["w"])]
        if not lo_idxs or not hi_idxs or not twin_idxs or not rotate_idxs:
            continue
        if not any(i != j for i in lo_idxs for j in hi_idxs):
            continue

        cavities = [{"h": int(info["h"]), "w": int(info["w"])} for info in selected]
        try:
            frame, placed_cavities = _build_frame(cavities)
        except ValueError:
            continue

        lo_idx = random.choice(lo_idxs)
        hi_idx_choices = [idx for idx in hi_idxs if idx != lo_idx]
        if not hi_idx_choices:
            continue
        hi_idx = random.choice(hi_idx_choices)

        rotate_idx = random.choice(rotate_idxs)
        twin_non_rotate = [idx for idx in twin_idxs if idx != rotate_idx]
        twin_idx = random.choice(twin_non_rotate or twin_idxs)

        areas = []
        for idx, info in enumerate(selected):
            choices = list(info["pass_areas"])
            if idx == lo_idx:
                areas.append(lo)
            elif idx == hi_idx:
                areas.append(hi)
            else:
                areas.append(int(random.choice(choices)))

        kept_specs: list[dict[str, object]] = []
        for idx, cavity in enumerate(placed_cavities):
            color = 0
            kept_specs.append(
                {
                    "cavity_idx": idx,
                    "cavity": cavity,
                    "area": areas[idx],
                    "rotate": idx == rotate_idx,
                    "color": color,
                }
            )

        twin_info = selected[twin_idx]
        out_areas = list(twin_info["out_areas"])
        preferred = [area for area in out_areas if area in (lo - 1, hi + 1)]
        rejected_area = int(random.choice(preferred or out_areas))

        return frame, placed_cavities, kept_specs, twin_idx, rejected_area
    return None


def _reserve_marker_cells(
    h: int,
    w: int,
    orange_count: int,
    teal_count: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    top_right = [(r, c) for r in range(MARKER_BOX) for c in range(w - MARKER_BOX, w)]
    bottom_left = [(r, c) for r in range(h - MARKER_BOX, h) for c in range(MARKER_BOX)]
    random.shuffle(top_right)
    random.shuffle(bottom_left)
    orange_cells = top_right[:orange_count]
    teal_cells = bottom_left[:teal_count]
    reserved_rects = [
        (0, w - MARKER_BOX, MARKER_BOX - 1, w - 1),
        (h - MARKER_BOX, 0, h - 1, MARKER_BOX - 1),
    ]
    return orange_cells, teal_cells, reserved_rects


def _sprite_positions(
    h: int,
    w: int,
    template_rect: tuple[int, int, int, int],
    reserved_rects: list[tuple[int, int, int, int]],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda idx: -(patches[idx].shape[0] * patches[idx].shape[1]))
    placed_rects: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _rect_conflicts(rect, template_rect, gap=1):
                    continue
                if any(_rect_conflicts(rect, other, gap=1) for other in placed_rects):
                    continue
                if any(_rect_conflicts(rect, other, gap=0) for other in reserved_rects):
                    continue
                candidates.append(rect)
        if not candidates:
            return None
        rect = random.choice(candidates)
        placed_rects.append(rect)
        positions[idx] = (rect[0], rect[1])

    assert all(pos is not None for pos in positions)
    return [(int(pos[0]), int(pos[1])) for pos in positions if pos is not None]


def _expected_output(frame: np.ndarray, kept_specs: list[dict[str, object]], patches: list[np.ndarray]) -> np.ndarray:
    out = np.array(frame, copy=True)
    for spec, patch in zip(kept_specs, patches):
        cavity = spec["cavity"]
        sprite = np.rot90(patch, 1) if bool(spec["rotate"]) else patch
        _place_patch(out, sprite, int(cavity["r0"]), int(cavity["c0"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    chosen = _choose_layout()
    if chosen is None:
        return None
    frame, cavities, kept_specs, twin_idx, rejected_area = chosen

    lo = min(int(spec["area"]) for spec in kept_specs)
    hi = max(int(spec["area"]) for spec in kept_specs)
    assert lo < hi

    total_sprites = len(kept_specs) + 1
    colors = random.sample(list(SPRITE_COLORS), total_sprites)

    patches: list[np.ndarray] = []
    for spec, color in zip(kept_specs, colors[:-1]):
        cavity = spec["cavity"]
        rotate = bool(spec["rotate"])
        sh = int(cavity["w"]) if rotate else int(cavity["h"])
        sw = int(cavity["h"]) if rotate else int(cavity["w"])
        patch = _sprite_patch(sh, sw, int(spec["area"]), color)
        spec["color"] = color
        patches.append(patch)

    twin_cavity = cavities[twin_idx]
    rejected_patch = _sprite_patch(int(twin_cavity["h"]), int(twin_cavity["w"]), rejected_area, colors[-1])
    rejected_spec = {
        "kind": "rejected",
        "cavity_idx": twin_idx,
        "bbox": (int(twin_cavity["h"]), int(twin_cavity["w"])),
        "area": rejected_area,
        "color": colors[-1],
    }

    expected = _expected_output(frame, kept_specs, patches)

    frame_h, frame_w = frame.shape
    template_r0 = 5
    template_c0 = 5
    grid_h = max(22, frame_h + template_r0 + MARKER_BOX + 2) + random.randint(0, 2)
    grid_w = max(22, frame_w + template_c0 + MARKER_BOX + 2) + random.randint(0, 2)
    if grid_h > 30 or grid_w > 30:
        return None

    template_rect = (template_r0, template_c0, template_r0 + frame_h - 1, template_c0 + frame_w - 1)
    if template_rect[2] >= grid_h or template_rect[3] >= grid_w:
        return None

    if random.random() < 0.5:
        orange_count, teal_count = lo, hi
    else:
        orange_count, teal_count = hi, lo
    orange_cells, teal_cells, reserved_rects = _reserve_marker_cells(grid_h, grid_w, orange_count, teal_count)

    all_patches = patches + [rejected_patch]
    positions = _sprite_positions(grid_h, grid_w, template_rect, reserved_rects, all_patches)
    if positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[template_r0 : template_r0 + frame_h, template_c0 : template_c0 + frame_w] = frame

    for patch, (r0, c0) in zip(all_patches, positions):
        _place_patch(grid, patch, r0, c0)

    for r, c in orange_cells:
        grid[r, c] = ORANGE
    for r, c in teal_cells:
        grid[r, c] = TEAL

    kept_dim_keys = {tuple(sorted((int(spec["cavity"]["h"]), int(spec["cavity"]["w"])))) for spec in kept_specs}
    assert len(kept_dim_keys) == len(kept_specs)
    assert any(bool(spec["rotate"]) for spec in kept_specs)
    assert rejected_spec["bbox"] in {(int(spec["cavity"]["h"]), int(spec["cavity"]["w"])) for spec in kept_specs}
    assert rejected_spec["area"] < lo or rejected_spec["area"] > hi
    assert len(all_patches) > len(cavities)
    assert len(kept_specs) == len(cavities)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
