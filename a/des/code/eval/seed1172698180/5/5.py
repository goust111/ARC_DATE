from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))
GREY = 5
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _bbox_from_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_list(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _frame_score(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return min(
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    )


def _is_target_frame(comp: dict, grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    if int(comp["area"]) != 2 * (height + width) - 4:
        return False

    local = np.zeros((height, width), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    local[rel[:, 0], rel[:, 1]] = True

    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != int(bg)):
        return False
    return True


def _find_target_frame(grid: np.ndarray, bg: int) -> dict:
    candidates = [comp for comp in _component_list(grid, bg) if _is_target_frame(comp, grid, bg)]
    assert candidates
    candidates.sort(key=lambda comp: (_frame_score(comp["bbox"], grid.shape), -int(comp["area"])))
    best = candidates[0]
    if len(candidates) > 1:
        best_key = (_frame_score(best["bbox"], grid.shape), -int(best["area"]))
        second_key = (_frame_score(candidates[1]["bbox"], grid.shape), -int(candidates[1]["area"]))
        assert best_key != second_key
    return best


def _leaf_pixels(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    leaves: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        deg = 0
        for dr, dc in DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                deg += 1
        if deg == 1:
            leaves.append((int(r), int(c)))
    return leaves


def _fragment_info(comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    mask = np.zeros((height, width), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    leaves = _leaf_pixels(mask)
    clean = mask.copy()
    for r, c in leaves:
        clean[r, c] = False
    assert bool(clean.any())
    clean = _crop_mask(clean)
    return {
        "color": int(comp["color"]),
        "step_count": int(len(leaves)),
        "clean_mask": clean,
        "bbox": comp["bbox"],
    }


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def _render_layers(
    shape: tuple[int, int],
    bg: int,
    frame: dict,
    layers: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    _draw_frame(out, frame["bbox"], int(frame["color"]))
    fr0, fc0, fr1, fc1 = frame["bbox"]
    inner_h = int(fr1 - fr0 - 1)
    inner_w = int(fc1 - fc0 - 1)

    for color, mask in layers:
        h, w = mask.shape
        assert h <= inner_h and w <= inner_w
        rr, cc = np.nonzero(mask)
        out[fr0 + 1 + rr, fc0 + 1 + cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered monochrome fragments inside the corner frame by using their leaf-pixel counts as layer order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frame = _find_target_frame(g, bg)
    frame_key = (int(frame["color"]), tuple(int(v) for v in frame["bbox"]), int(frame["area"]))
    fragments = [
        _fragment_info(comp)
        for comp in _component_list(g, bg)
        if (int(comp["color"]), tuple(int(v) for v in comp["bbox"]), int(comp["area"])) != frame_key
    ]
    assert fragments

    ordered = sorted(
        fragments,
        key=lambda item: (int(item["step_count"]), int(item["bbox"][0]), int(item["bbox"][1])),
    )
    layers = [(int(item["color"]), item["clean_mask"]) for item in ordered]
    return _render_layers(g.shape, bg, frame, layers)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _leafless(mask: np.ndarray) -> bool:
    if not _mask_connected(mask):
        return False
    return len(_leaf_pixels(mask)) == 0


def _sample_clean_mask(height: int, width: int) -> np.ndarray:
    for _ in range(80):
        mode = random.choice(("solid", "corner_notch", "side_notch"))
        mask = np.ones((height, width), dtype=bool)

        if mode == "corner_notch":
            if height < 4 or width < 4:
                continue
            notch_h = random.randint(1, height - 2)
            notch_w = random.randint(1, width - 2)
            if height - notch_h < 2 or width - notch_w < 2:
                continue
            corner = random.choice(("tl", "tr", "bl", "br"))
            if corner == "tl":
                mask[:notch_h, :notch_w] = False
            elif corner == "tr":
                mask[:notch_h, width - notch_w :] = False
            elif corner == "bl":
                mask[height - notch_h :, :notch_w] = False
            else:
                mask[height - notch_h :, width - notch_w :] = False
        elif mode == "side_notch":
            if height < 5 or width < 5:
                continue
            side = random.choice(("u", "d", "l", "r"))
            notch_h = random.randint(1, height - 2)
            notch_w = random.randint(1, width - 2)
            if height - notch_h < 2 or width - notch_w < 2:
                continue
            if side in ("u", "d"):
                if width - notch_w < 2:
                    continue
                start_c = random.randint(1, width - notch_w - 1)
                rows = slice(0, notch_h) if side == "u" else slice(height - notch_h, height)
                mask[rows, start_c : start_c + notch_w] = False
            else:
                if height - notch_h < 2:
                    continue
                start_r = random.randint(1, height - notch_h - 1)
                cols = slice(0, notch_w) if side == "l" else slice(width - notch_w, width)
                mask[start_r : start_r + notch_h, cols] = False

        if _leafless(mask):
            r0, c0, r1, c1 = _bbox_from_mask(mask)
            if (r0, c0, r1, c1) == (0, 0, height - 1, width - 1):
                return mask

    return np.ones((height, width), dtype=bool)


def _outer_step_candidates(clean_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = clean_mask.shape
    padded = np.pad(clean_mask, 1, constant_values=False)
    candidates: list[tuple[int, int]] = []

    for r in range(h + 2):
        for c in range(w + 2):
            if bool(padded[r, c]):
                continue
            if not (r == 0 or r == h + 1 or c == 0 or c == w + 1):
                continue
            deg = 0
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h + 2 and 0 <= nc < w + 2 and bool(padded[nr, nc]):
                    deg += 1
            if deg == 1:
                candidates.append((r, c))
    return candidates


def _attach_steps(clean_mask: np.ndarray, count: int) -> np.ndarray | None:
    padded = np.pad(clean_mask, 1, constant_values=False)
    candidates = _outer_step_candidates(clean_mask)
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == int(count):
            break

    if len(chosen) != int(count):
        return None

    full = padded.copy()
    for r, c in chosen:
        full[r, c] = True
    if len(_leaf_pixels(full)) != int(count):
        return None
    return _crop_mask(full)


def _visible_layer_masks(masks: list[np.ndarray], inner_h: int, inner_w: int) -> bool:
    cover = np.zeros((inner_h, inner_w), dtype=bool)
    for mask in reversed(masks):
        h, w = mask.shape
        visible = np.zeros((inner_h, inner_w), dtype=bool)
        visible[:h, :w] = mask
        if not bool((visible & ~cover).any()):
            return False
        cover |= visible
    return True


def _has_overlap(masks: list[np.ndarray], inner_h: int, inner_w: int) -> bool:
    counts = np.zeros((inner_h, inner_w), dtype=int)
    for mask in masks:
        h, w = mask.shape
        counts[:h, :w] += mask.astype(int)
    return int(counts.max()) >= 2


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _frame_bbox_at_corner(
    grid_h: int,
    grid_w: int,
    frame_h: int,
    frame_w: int,
    corner: str,
) -> tuple[int, int, int, int]:
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, grid_w - frame_w
    elif corner == "bl":
        r0, c0 = grid_h - frame_h, 0
    else:
        r0, c0 = grid_h - frame_h, grid_w - frame_w
    return r0, c0, r0 + frame_h - 1, c0 + frame_w - 1


def _paint_sprite(grid: np.ndarray, sprite: np.ndarray, color: int, top: int, left: int) -> None:
    rr, cc = np.nonzero(sprite)
    grid[top + rr, left + cc] = int(color)


def _sprite_coords(sprite: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rr, cc = np.nonzero(sprite)
    return [(top + int(r), left + int(c)) for r, c in zip(rr, cc)]


def _place_sprites(
    grid: np.ndarray,
    blocked: np.ndarray,
    sprites: list[tuple[int, np.ndarray]],
) -> bool:
    h, w = grid.shape
    items = list(sprites)
    random.shuffle(items)

    for color, sprite in items:
        sh, sw = sprite.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - sh)
            left = random.randint(0, w - sw)
            coords = _sprite_coords(sprite, top, left)
            if any(bool(blocked[r, c]) for r, c in coords):
                continue
            _paint_sprite(grid, sprite, color, top, left)
            _mark_blocked(blocked, coords)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected(
    shape: tuple[int, int],
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
    layers: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    frame = {"bbox": frame_bbox, "color": int(frame_color)}
    return _render_layers(shape, bg, frame, layers)


def _generate_attempt() -> np.ndarray | None:
    n_fragments = random.randint(2, 4)
    min_inner_h = max(5, n_fragments + 2)
    min_inner_w = max(5, n_fragments + 2)
    inner_h = random.randint(min_inner_h, 9)
    inner_w = random.randint(min_inner_w, 9)

    heights = sorted(random.sample(range(2, inner_h + 1), n_fragments), reverse=True)
    widths = sorted(random.sample(range(2, inner_w + 1), n_fragments), reverse=True)
    clean_masks = [_sample_clean_mask(h, w) for h, w in zip(heights, widths)]

    if not _has_overlap(clean_masks, inner_h, inner_w):
        return None
    if not _visible_layer_masks(clean_masks, inner_h, inner_w):
        return None

    capacities = [len(_outer_step_candidates(mask)) for mask in clean_masks]
    if min(capacities) < n_fragments:
        return None
    counts = list(range(1, n_fragments + 1))

    sprites: list[np.ndarray] = []
    for mask, count in zip(clean_masks, counts):
        sprite = _attach_steps(mask, count)
        if sprite is None:
            return None
        sprites.append(sprite)

    bg_choices = [color for color in PALETTE if color != GREY]
    bg = int(random.choice(bg_choices))
    piece_colors = random.sample([color for color in PALETTE if color not in (bg, GREY)], n_fragments)

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    max_sprite_h = max(sprite.shape[0] for sprite in sprites)
    max_sprite_w = max(sprite.shape[1] for sprite in sprites)

    min_grid_h = max(frame_h + max_sprite_h + 3, 13)
    min_grid_w = max(frame_w + max_sprite_w + 3, 13)
    if min_grid_h > 24 or min_grid_w > 24:
        return None
    grid_h = random.randint(min_grid_h, min(24, min_grid_h + 6))
    grid_w = random.randint(min_grid_w, min(24, min_grid_w + 6))

    corner = random.choice(("tl", "tr", "bl", "br"))
    frame_bbox = _frame_bbox_at_corner(grid_h, grid_w, frame_h, frame_w, corner)

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    r0, c0, r1, c1 = frame_bbox
    frame_coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_blocked(blocked, frame_coords)
    _draw_frame(inp, frame_bbox, GREY)

    fragment_sprites = list(zip(piece_colors, sprites))
    if not _place_sprites(inp, blocked, fragment_sprites):
        return None

    layers = [(int(color), mask) for color, mask in zip(piece_colors, clean_masks)]
    expected = _build_expected(inp.shape, bg, frame_bbox, GREY, layers)

    if np.count_nonzero(inp == bg) <= np.count_nonzero(inp != bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
