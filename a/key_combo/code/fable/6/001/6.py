from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            out.append(np.array(coords, dtype=int))

    return out


def find_components(grid: np.ndarray, *, bg: int | None = None) -> list[dict]:
    if bg is None:
        bg = get_bg(grid)
    comps = []
    for coords in components_mask(grid != int(bg)):
        comps.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _top_view(crop: np.ndarray, bg: int) -> tuple[int, ...]:
    h, w = crop.shape
    seq = []
    for c in range(w):
        rows = np.flatnonzero(crop[:, c] != int(bg))
        assert rows.size > 0
        seq.append(int(crop[int(rows[0]), c]))
    return tuple(seq)


def _left_view(crop: np.ndarray, bg: int) -> tuple[int, ...]:
    h, w = crop.shape
    seq = []
    for r in range(h):
        cols = np.flatnonzero(crop[r, :] != int(bg))
        assert cols.size > 0
        seq.append(int(crop[r, int(cols[0])]))
    return tuple(seq)


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    components = find_components(g, bg=bg)
    assert len(components) >= 5

    horizontal_ids = [
        idx
        for idx, comp in enumerate(components)
        if bbox_h(comp["bbox"]) == 1 and int(comp["area"]) == bbox_w(comp["bbox"]) and bbox_w(comp["bbox"]) >= 2
    ]
    vertical_ids = [
        idx
        for idx, comp in enumerate(components)
        if bbox_w(comp["bbox"]) == 1 and int(comp["area"]) == bbox_h(comp["bbox"]) and bbox_h(comp["bbox"]) >= 2
    ]
    assert len(horizontal_ids) == 1
    assert len(vertical_ids) == 1
    h_idx = int(horizontal_ids[0])
    v_idx = int(vertical_ids[0])
    assert h_idx != v_idx

    top_strip = crop_bbox(g, components[h_idx]["bbox"])
    left_strip = crop_bbox(g, components[v_idx]["bbox"])
    top_seq = tuple(int(x) for x in top_strip[0, :])
    left_seq = tuple(int(x) for x in left_strip[:, 0])

    candidates = []
    for idx, comp in enumerate(components):
        if idx in (h_idx, v_idx):
            continue

        crop = crop_bbox(g, comp["bbox"])
        match_top = False
        match_left = False
        top_view = None
        left_view = None

        if crop.shape == (len(left_seq), len(top_seq)):
            top_view = _top_view(crop, bg)
            left_view = _left_view(crop, bg)
            match_top = top_view == top_seq
            match_left = left_view == left_seq

        candidates.append(
            {
                "bbox": comp["bbox"],
                "crop": crop,
                "top_view": top_view,
                "left_view": left_view,
                "match_top": bool(match_top),
                "match_left": bool(match_left),
            }
        )

    assert candidates
    return {
        "bg": bg,
        "top_seq": top_seq,
        "left_seq": left_seq,
        "candidates": candidates,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the unique candidate sprite whose top-view strip and left-view strip both match the two separate legends.

    Input structure:
    - Black background surrounds one separated horizontal color strip near the top and one separated vertical color strip near the left.
    - The remaining foreground objects are multicolor candidate sprites, each considered inside its own tight bounding box.
    - The target sprite shares the horizontal strip's width and the vertical strip's height; distractors keep the same dimensions.

    Rule:
    - Read the horizontal strip as the top-view sequence: for each candidate column, compare the color of the first non-background pixel seen from top to bottom.
    - Read the vertical strip as the left-view sequence: for each candidate row, compare the color of the first non-background pixel seen from left to right.
    - Find the unique candidate whose two occlusion-view sequences both match the two strips, then output that candidate's tight bounding box crop.

    Key insight:
    - Each strip records only the first visible color per column or per row, not a border, count, or silhouette, so distractors can match one side while failing the other.
    """
    scene = _scene_info(grid)
    matches = [
        candidate
        for candidate in scene["candidates"]
        if candidate["match_top"] and candidate["match_left"]
    ]
    assert len(matches) == 1
    return matches[0]["crop"]


def _different_color(avoid: int, palette: list[int]) -> int:
    choices = [color for color in palette if int(color) != int(avoid)]
    if not choices:
        choices = [color for color in COLORS if int(color) != int(avoid)]
    return int(random.choice(choices))


def _mask_from_left_profile(left_profile: tuple[int, ...], width: int) -> np.ndarray:
    mask = np.zeros((len(left_profile), int(width)), dtype=bool)
    for r, left in enumerate(left_profile):
        mask[int(r), int(left) : int(width)] = True
    return mask


def _left_profile(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(np.flatnonzero(mask[r, :])[0]) for r in range(mask.shape[0]))


def _top_profile(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(np.flatnonzero(mask[:, c])[0]) for c in range(mask.shape[1]))


def _sample_mask(height: int, width: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], list[int], list[int]] | None:
    area_min = int(np.ceil(0.48 * int(height) * int(width)))
    area_max = int(np.floor(0.82 * int(height) * int(width)))

    for _ in range(120):
        start = random.randint(1, int(width) - 2)
        middle = [random.randint(0, start) for _ in range(max(0, int(height) - 2))]
        left = (start, *sorted(middle, reverse=True), 0)
        mask = _mask_from_left_profile(tuple(int(x) for x in left), int(width))
        area = int(mask.sum())
        if not (area_min <= area <= area_max):
            continue

        top = _top_profile(mask)
        left = _left_profile(mask)
        if not any(value > 0 for value in top):
            continue
        if not any(value > 0 for value in left):
            continue
        if len(set(top)) < 2 or len(set(left)) < 2:
            continue

        top_positions = {(int(top[c]), int(c)) for c in range(width)}
        left_positions = {(int(r), int(left[r])) for r in range(height)}
        pure_left_rows = [int(r) for r in range(height) if (int(r), int(left[r])) not in top_positions]
        pure_top_cols = [int(c) for c in range(width) if (int(top[c]), int(c)) not in left_positions]
        if not pure_left_rows or not pure_top_cols:
            continue

        return mask, left, top, pure_left_rows, pure_top_cols

    return None


def _random_sprite(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    coords = np.argwhere(mask)
    for r, c in coords:
        sprite[int(r), int(c)] = int(random.choice(palette))
    return sprite


def _sample_target_sprite(mask: np.ndarray, palette: list[int]) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]] | None:
    for _ in range(120):
        sprite = _random_sprite(mask, palette)
        top_seq = _top_view(sprite, BG)
        left_seq = _left_view(sprite, BG)
        if len(set(top_seq)) < 2 or len(set(left_seq)) < 2:
            continue
        if len(set(top_seq + left_seq)) < 3:
            continue
        return sprite, top_seq, left_seq
    return None


def _build_guided_sprite(
    mask: np.ndarray,
    palette: list[int],
    left: tuple[int, ...],
    top: tuple[int, ...],
    left_seq: tuple[int, ...],
    top_seq: tuple[int, ...],
    pure_left_rows: list[int],
    pure_top_cols: list[int],
    mode: str,
) -> np.ndarray:
    sprite = _random_sprite(mask, palette)

    for r, c in enumerate(left):
        sprite[int(r), int(c)] = int(left_seq[r])
    for c, r in enumerate(top):
        sprite[int(r), int(c)] = int(top_seq[c])

    if mode == "target":
        return sprite

    if mode == "top_only":
        bad_row = int(random.choice(pure_left_rows))
        sprite[bad_row, int(left[bad_row])] = _different_color(int(left_seq[bad_row]), palette)
        return sprite

    if mode == "left_only":
        bad_col = int(random.choice(pure_top_cols))
        sprite[int(top[bad_col]), bad_col] = _different_color(int(top_seq[bad_col]), palette)
        return sprite

    if mode == "none":
        bad_row = int(random.choice(pure_left_rows))
        sprite[bad_row, int(left[bad_row])] = _different_color(int(left_seq[bad_row]), palette)
        bad_col_choices = [
            int(col)
            for col in pure_top_cols
            if (int(top[col]), int(col)) != (bad_row, int(left[bad_row]))
        ]
        bad_col = int(random.choice(bad_col_choices or pure_top_cols))
        sprite[int(top[bad_col]), bad_col] = _different_color(int(top_seq[bad_col]), palette)
        return sprite

    raise ValueError(f"unknown mode: {mode}")


def _sample_alt_none_sprite(height: int, width: int, palette: list[int], top_seq: tuple[int, ...], left_seq: tuple[int, ...]) -> np.ndarray | None:
    for _ in range(120):
        sampled = _sample_mask(int(height), int(width))
        if sampled is None:
            continue
        mask2, _, _, _, _ = sampled
        sprite = _random_sprite(mask2, palette)
        if _top_view(sprite, BG) != top_seq and _left_view(sprite, BG) != left_seq:
            return sprite
    return None


def _assemble_grid(candidates: list[np.ndarray], top_seq: tuple[int, ...], left_seq: tuple[int, ...]) -> np.ndarray:
    height, width = map(int, candidates[0].shape)
    n_candidates = len(candidates)
    if n_candidates == 4:
        layout_rows, layout_cols = 2, 2
        slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        layout_rows, layout_cols = 2, 3
        all_slots = [(r, c) for r in range(2) for c in range(3)]
        slots = random.sample(all_slots, n_candidates)

    top_strip_r = random.randint(1, 2)
    top_strip_c = random.randint(3, 5)
    left_strip_r = random.randint(3, 4)
    left_strip_c = random.randint(1, 2)
    row_gap = random.randint(2, 3)
    col_gap = random.randint(2, 3)
    candidate_r0 = int(left_strip_r + height + random.randint(2, 3))
    candidate_c0 = random.randint(3, 4)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    candidate_bottom = candidate_r0 + layout_rows * height + (layout_rows - 1) * row_gap
    candidate_right = candidate_c0 + layout_cols * width + (layout_cols - 1) * col_gap
    grid_h = max(int(left_strip_r + height), int(candidate_bottom)) + bottom_margin
    grid_w = max(int(top_strip_c + width), int(candidate_right)) + right_margin
    assert grid_h <= 30 and grid_w <= 30

    grid = np.zeros((int(grid_h), int(grid_w)), dtype=int)
    paste(grid, np.array([top_seq], dtype=int), int(top_strip_r), int(top_strip_c))
    paste(grid, np.array(left_seq, dtype=int).reshape(-1, 1), int(left_strip_r), int(left_strip_c))

    shuffled = candidates[:]
    random.shuffle(shuffled)
    for sprite, (slot_r, slot_c) in zip(shuffled, slots):
        r0 = int(candidate_r0 + slot_r * (height + row_gap))
        c0 = int(candidate_c0 + slot_c * (width + col_gap))
        paste(grid, sprite, r0, c0)

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    scene = _scene_info(grid)
    top_seq = scene["top_seq"]
    left_seq = scene["left_seq"]
    candidates = scene["candidates"]
    assert len(candidates) >= 4
    assert all(candidate["crop"].shape == (len(left_seq), len(top_seq)) for candidate in candidates)

    top_matches = [candidate for candidate in candidates if candidate["match_top"]]
    left_matches = [candidate for candidate in candidates if candidate["match_left"]]
    both_matches = [
        candidate
        for candidate in candidates
        if candidate["match_top"] and candidate["match_left"]
    ]

    assert len(top_matches) >= 2
    assert len(left_matches) >= 2
    assert len(both_matches) == 1
    assert any(candidate["match_top"] and not candidate["match_left"] for candidate in candidates)
    assert any(candidate["match_left"] and not candidate["match_top"] for candidate in candidates)

    target = both_matches[0]["crop"]
    assert np.any(target == BG)
    target_top = _top_profile(target != BG)
    target_left = _left_profile(target != BG)
    assert any(value > 0 for value in target_top)
    assert any(value > 0 for value in target_left)
    assert np.array_equal(main(grid), target)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(4, 6)
    width = random.randint(4, 6)
    sampled = _sample_mask(height, width)
    if sampled is None:
        return None
    mask, left, top, pure_left_rows, pure_top_cols = sampled

    palette_size = random.randint(3, 5)
    palette = random.sample(list(COLORS), palette_size)
    target_sample = _sample_target_sprite(mask, palette)
    if target_sample is None:
        return None
    _target_seed, top_seq, left_seq = target_sample

    target = _build_guided_sprite(
        mask, palette, left, top, left_seq, top_seq, pure_left_rows, pure_top_cols, "target"
    )
    top_only = _build_guided_sprite(
        mask, palette, left, top, left_seq, top_seq, pure_left_rows, pure_top_cols, "top_only"
    )
    left_only = _build_guided_sprite(
        mask, palette, left, top, left_seq, top_seq, pure_left_rows, pure_top_cols, "left_only"
    )
    none_same = _build_guided_sprite(
        mask, palette, left, top, left_seq, top_seq, pure_left_rows, pure_top_cols, "none"
    )

    candidates = [target, top_only, left_only, none_same]
    if random.random() < 0.65:
        none_alt = _sample_alt_none_sprite(height, width, palette, top_seq, left_seq)
        if none_alt is not None:
            candidates.append(none_alt)

    grid = _assemble_grid(candidates, top_seq, left_seq)
    _validate_instance(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-occlusion-view input")
