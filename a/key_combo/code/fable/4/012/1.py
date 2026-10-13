from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(arr)

    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray | None:
    rr, cc = np.nonzero(mask)
    rr2 = rr + int(dr)
    cc2 = cc + int(dc)
    h, w = mask.shape
    if rr.size == 0:
        return np.zeros_like(mask, dtype=bool)
    if np.any(rr2 < 0) or np.any(rr2 >= h) or np.any(cc2 < 0) or np.any(cc2 >= w):
        return None
    out = np.zeros_like(mask, dtype=bool)
    out[rr2, cc2] = True
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _local_visible_coords(
    body_mask: np.ndarray,
    shadow_mask: np.ndarray,
    coords: np.ndarray,
    dr: int,
    dc: int,
) -> np.ndarray | None:
    shifted = coords + np.array([int(dr), int(dc)], dtype=int)
    h, w = body_mask.shape
    if (
        np.any(shifted[:, 0] < 0)
        or np.any(shifted[:, 0] >= h)
        or np.any(shifted[:, 1] < 0)
        or np.any(shifted[:, 1] >= w)
    ):
        return None

    r0 = int(min(coords[:, 0].min(), shifted[:, 0].min()))
    c0 = int(min(coords[:, 1].min(), shifted[:, 1].min()))
    r1 = int(max(coords[:, 0].max(), shifted[:, 0].max()))
    c1 = int(max(coords[:, 1].max(), shifted[:, 1].max()))

    body_region = body_mask[r0 : r1 + 1, c0 : c1 + 1]
    shadow_region = shadow_mask[r0 : r1 + 1, c0 : c1 + 1]

    comp_local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    comp_local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    visible = ~body_mask[shifted[:, 0], shifted[:, 1]]
    if not bool(visible.any()):
        return None

    pred_local = np.zeros_like(comp_local)
    pred_local[shifted[visible, 0] - r0, shifted[visible, 1] - c0] = True

    if np.array_equal(body_region, comp_local) and np.array_equal(shadow_region, pred_local):
        return shifted[visible]
    return None


def _full_configuration_valid(
    body_mask: np.ndarray,
    shadow_mask: np.ndarray,
    missing_body: np.ndarray,
    dr: int,
    dc: int,
) -> bool:
    completed_body = body_mask | missing_body
    completed_shadow = shadow_mask & ~missing_body
    predicted_shadow = np.zeros_like(shadow_mask, dtype=bool)

    for coords in _components(completed_body, connectivity=8):
        visible_coords = _local_visible_coords(completed_body, completed_shadow, coords, dr, dc)
        if visible_coords is None:
            return False
        predicted_shadow[visible_coords[:, 0], visible_coords[:, 1]] = True

    return np.array_equal(predicted_shadow, completed_shadow)


def _candidate_vectors(body_mask: np.ndarray, shadow_mask: np.ndarray) -> list[tuple[int, int]]:
    body_pts = np.argwhere(body_mask)
    shadow_pts = np.argwhere(shadow_mask)
    vectors = {
        (int(sr - br), int(sc - bc))
        for br, bc in body_pts
        for sr, sc in shadow_pts
        if (int(sr - br), int(sc - bc)) != (0, 0)
    }
    return sorted(vectors, key=lambda v: (abs(v[0]) + abs(v[1]), abs(v[0]), abs(v[1]), v[0], v[1]))


def _evaluate_candidate(
    body_mask: np.ndarray,
    shadow_mask: np.ndarray,
    dr: int,
    dc: int,
) -> tuple[tuple[int, int, int], np.ndarray] | None:
    explained_shadow = np.zeros_like(shadow_mask, dtype=bool)
    support_count = 0

    for coords in _components(body_mask, connectivity=8):
        visible_coords = _local_visible_coords(body_mask, shadow_mask, coords, dr, dc)
        if visible_coords is None:
            continue
        support_count += 1
        explained_shadow[visible_coords[:, 0], visible_coords[:, 1]] = True

    if support_count == 0:
        return None

    leftover_shadow = shadow_mask & ~explained_shadow
    if not bool(leftover_shadow.any()):
        return None
    if len(_components(leftover_shadow, connectivity=8)) != 1:
        return None

    missing_body = _shift_mask(leftover_shadow, -dr, -dc)
    if missing_body is None or np.any(missing_body & body_mask):
        return None
    if not _full_configuration_valid(body_mask, shadow_mask, missing_body, dr, dc):
        return None

    score = (support_count, int(explained_shadow.sum()), int(missing_body.sum()))
    return score, missing_body


def _infer_repair(grid: np.ndarray) -> tuple[int, np.ndarray]:
    bg = _get_bg(grid)
    colors = _colors_present(grid, exclude={bg})
    assert len(colors) == 2, "expected exactly one body color and one shadow color"

    best: tuple[tuple[int, int, int], int, np.ndarray] | None = None
    for body_color, shadow_color in ((colors[0], colors[1]), (colors[1], colors[0])):
        body_mask = grid == int(body_color)
        shadow_mask = grid == int(shadow_color)
        if not bool(body_mask.any()) or not bool(shadow_mask.any()):
            continue
        for dr, dc in _candidate_vectors(body_mask, shadow_mask):
            evaluated = _evaluate_candidate(body_mask, shadow_mask, dr, dc)
            if evaluated is None:
                continue
            score, missing_body = evaluated
            if best is None or score > best[0]:
                best = (score, int(body_color), missing_body)

    assert best is not None, "could not infer the common shadow offset"
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the lone full grey shadow and restore its missing same-color sprite by reversing the global shadow offset.

    Input structure:
    - A dominant background contains several disconnected same-color foreground sprites and grey shadow pixels.
    - Each intact sprite has a shadow made by translating the sprite by one common offset vector, then hiding the overlap under the later-drawn sprite body.
    - Exactly one sprite body is missing, so its translated grey shadow remains fully visible.

    Rule:
    - Determine which foreground color is the sprite body and which is the grey shadow, and infer the unique global offset vector from any intact body-plus-residue pair where translated body minus body equals the local grey residue.
    - Use that offset to identify the shadow pixels not explained by existing bodies; this unexplained connected grey shape is the orphan full shadow.
    - Translate that orphan shadow back by the opposite vector and paint the recovered cells with the sprite color, leaving all other pixels unchanged.

    Key insight:
    - The grey crescent is not an independent object: it is the visible part of a translated copy of the sprite, so the one complete grey copy directly reveals the erased body when shifted back.
    """
    g = np.asarray(grid, dtype=int)
    body_color, missing_body = _infer_repair(g)
    out = g.copy()
    out[missing_body] = int(body_color)
    return out


_SHADOW_COLOR = 5
_MAX_ATTEMPTS = 400
_OFFSET_CHOICES = [(dr, dc) for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2) if (dr, dc) != (0, 0)]


def _random_blob_mask() -> np.ndarray:
    for _ in range(80):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        min_area = max(5, int(np.ceil(0.45 * h * w)))
        max_area = max(min_area, int(np.floor(0.75 * h * w)))
        target = random.randint(min_area, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = start[0] + dr
            nc = start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            rr, cc = cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        trimmed = _trim_mask(mask)
        if trimmed.shape[0] >= 2 and trimmed.shape[1] >= 2 and int(trimmed.sum()) >= 5:
            return trimmed
    raise RuntimeError("failed to generate blob sprite")


def _random_notched_rect_mask() -> np.ndarray:
    for _ in range(80):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)
        cut_h = random.randint(1, h - 1)
        cut_w = random.randint(1, w - 1)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:cut_h, :cut_w] = False
        elif corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
        if int(mask.sum()) >= 5:
            return mask
    raise RuntimeError("failed to generate notched rectangle sprite")


def _random_sprite_mask() -> np.ndarray:
    if random.random() < 0.55:
        mask = _random_blob_mask()
    else:
        mask = _random_notched_rect_mask()

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.35:
        mask = np.rot90(mask, k=random.randint(1, 3))
    return mask.astype(bool, copy=True)


def _build_object_patches(
    sprite_mask: np.ndarray,
    dr: int,
    dc: int,
    body_color: int,
    bg: int,
    orphan: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = sprite_mask.shape
    top = min(0, int(dr))
    left = min(0, int(dc))
    patch_h = h + abs(int(dr))
    patch_w = w + abs(int(dc))

    body_r0 = -top
    body_c0 = -left
    shadow_r0 = body_r0 + int(dr)
    shadow_c0 = body_c0 + int(dc)

    body_patch = np.zeros((patch_h, patch_w), dtype=bool)
    shadow_patch = np.zeros((patch_h, patch_w), dtype=bool)
    body_patch[body_r0 : body_r0 + h, body_c0 : body_c0 + w] = sprite_mask
    shadow_patch[shadow_r0 : shadow_r0 + h, shadow_c0 : shadow_c0 + w] = sprite_mask

    render = np.full((patch_h, patch_w), int(bg), dtype=int)
    render[shadow_patch] = _SHADOW_COLOR
    if not orphan:
        render[body_patch] = int(body_color)

    occupancy = body_patch | shadow_patch
    missing_body = body_patch if orphan else np.zeros_like(body_patch, dtype=bool)
    return render, occupancy, missing_body


def _random_free_location(
    occupied: np.ndarray,
    patch_mask: np.ndarray,
    *,
    border: int = 1,
    padding: int = 1,
) -> Coord | None:
    h, w = occupied.shape
    ph, pw = patch_mask.shape
    r_min = int(border)
    c_min = int(border)
    r_max = h - int(border) - ph
    c_max = w - int(border) - pw
    if r_max < r_min or c_max < c_min:
        return None

    forbidden = occupied.copy()
    if padding > 0 and bool(occupied.any()):
        for rr, cc in np.argwhere(occupied):
            r0 = max(0, int(rr) - int(padding))
            c0 = max(0, int(cc) - int(padding))
            r1 = min(h, int(rr) + int(padding) + 1)
            c1 = min(w, int(cc) + int(padding) + 1)
            forbidden[r0:r1, c0:c1] = True

    candidates: list[Coord] = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            if not bool((forbidden[r0 : r0 + ph, c0 : c0 + pw] & patch_mask).any()):
                candidates.append((r0, c0))

    return random.choice(candidates) if candidates else None


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _paste_mask(dst: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch_mask.shape
    dst[r0 : r0 + h, c0 : c0 + w] |= patch_mask


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    bg = random.choice([c for c in range(10) if c != _SHADOW_COLOR])
    body_color = random.choice([c for c in range(1, 10) if c not in {bg, _SHADOW_COLOR}])
    dr, dc = random.choice(_OFFSET_CHOICES)

    intact_count = random.randint(2, 4)
    total_count = intact_count + 1
    orphan_index = random.randrange(total_count)

    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    missing_body = np.zeros((h, w), dtype=bool)

    for idx in range(total_count):
        sprite_mask = _random_sprite_mask()
        render, occupancy, missing_patch = _build_object_patches(
            sprite_mask,
            dr,
            dc,
            body_color,
            bg,
            orphan=(idx == orphan_index),
        )
        loc = _random_free_location(occupied, occupancy, border=1, padding=1)
        if loc is None:
            return None
        r0, c0 = loc
        _paste_patch(grid, render, r0, c0, bg)
        _paste_mask(occupied, occupancy, r0, c0)
        if idx == orphan_index:
            _paste_mask(missing_body, missing_patch, r0, c0)

    expected = grid.copy()
    expected[missing_body] = int(body_color)

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orphan-shadow input")
