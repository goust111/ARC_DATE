from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask.astype(bool))
    return tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes()


def _has_hole(mask: np.ndarray) -> bool:
    tight = _tight_mask(mask.astype(bool))
    bg = ~tight
    for comp in _components(bg):
        rr = comp[:, 0]
        cc = comp[:, 1]
        if rr.min() > 0 and cc.min() > 0 and rr.max() < tight.shape[0] - 1 and cc.max() < tight.shape[1] - 1:
            return True
    return False


def _extract_strip_key(
    grid: np.ndarray,
) -> tuple[int, int, int, list[np.ndarray], np.ndarray]:
    strip_rows = np.flatnonzero(np.any(grid == GREY, axis=1))
    assert strip_rows.size > 0
    assert np.array_equal(strip_rows, np.arange(strip_rows[0], strip_rows[-1] + 1))
    strip_top = int(strip_rows[0])
    strip_bottom = int(strip_rows[-1])

    band = grid[strip_top : strip_bottom + 1]
    divider_cols = [c for c in range(grid.shape[1]) if np.all(band[:, c] == GREY)]
    assert len(divider_cols) >= 2

    fragment_colors = sorted(set(map(int, np.unique(band))) - {BLACK, GREY})
    assert len(fragment_colors) == 1
    fragment_color = fragment_colors[0]

    piece_masks: list[np.ndarray] = []
    widths: list[int] = []
    for left, right in zip(divider_cols, divider_cols[1:]):
        if right - left <= 1:
            continue
        piece = band[:, left + 1 : right] == fragment_color
        piece_masks.append(piece)
        widths.append(int(right - left - 1))

    assert len(piece_masks) >= 2
    assert len(set(widths)) == 1

    key_mask = np.zeros_like(piece_masks[0], dtype=bool)
    for piece in piece_masks:
        key_mask |= piece
    assert key_mask.any()
    return strip_top, strip_bottom, fragment_color, piece_masks, key_mask


def _extract_lock_holes(top: np.ndarray) -> list[dict[str, np.ndarray]]:
    holes: list[dict[str, np.ndarray]] = []
    for obj_coords in _components(top != BLACK):
        r0, c0, r1, c1 = _bbox(obj_coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[obj_coords[:, 0] - r0, obj_coords[:, 1] - c0] = True

        for hole_local in _components(~local):
            rr = hole_local[:, 0]
            cc = hole_local[:, 1]
            if rr.min() == 0 or cc.min() == 0 or rr.max() == local.shape[0] - 1 or cc.max() == local.shape[1] - 1:
                continue
            hole_mask = np.zeros_like(local, dtype=bool)
            hole_mask[rr, cc] = True
            holes.append(
                {
                    "coords": hole_local + np.array([[r0, c0]], dtype=int),
                    "mask": _tight_mask(hole_mask),
                }
            )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the bottom fragments into one key, find the unique matching lock hole above, and fill it.

    Input structure:
    - The bottom strip is split by grey vertical divider lines into equal-width black compartments.
    - Each compartment contains same-color fragment pixels, and the aligned union of those fragments forms one key mask.
    - Above the strip sit separate colored objects whose enclosed black cavities act as lock holes.

    Rule:
    - Detect the bottom strip, recover the fragment color, and overlay all compartment masks in register to obtain the key shape.
    - For every object above the strip, extract each enclosed black cavity and compare its tight mask to the key using translation-only shape matching.
    - Fill the unique matching cavity with the fragment color and clear the entire bottom strip to black.

    Key insight:
    - The object to match is not any visible fragment or outer object contour; the decisive shape is the negative-space cavity whose mask equals the union of all bottom fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    strip_top, strip_bottom, fragment_color, _pieces, key_mask = _extract_strip_key(g)
    key_signature = _mask_signature(key_mask)

    matches = [hole for hole in _extract_lock_holes(g[:strip_top]) if _mask_signature(hole["mask"]) == key_signature]
    assert len(matches) == 1

    out = g.copy()
    hole_coords = matches[0]["coords"]
    out[hole_coords[:, 0], hole_coords[:, 1]] = fragment_color
    out[strip_top : strip_bottom + 1, :] = BLACK
    return out


def _random_connected_mask(*, exact_area: int | None = None) -> np.ndarray | None:
    for _ in range(120):
        if exact_area is None:
            h = random.randint(3, 6)
            w = random.randint(4, 6)
            lo = max(5, (h * w + 2) // 3)
            hi = max(lo, min(h * w - 1, (2 * h * w) // 3 + 1))
            area = random.randint(lo, hi)
        else:
            area = exact_area
            options = [(h, w) for h in range(3, 7) for w in range(3, 7) if area <= h * w - 1]
            if not options:
                return None
            h, w = random.choice(options)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()
        for dr, dc in NEIGHBORS4:
            nr = start[0] + dr
            nc = start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while int(mask.sum()) < area and frontier:
            choices = list(frontier)
            weights = []
            for r, c in choices:
                touch = sum(
                    0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in NEIGHBORS4
                )
                weights.append(1 + touch * touch)
            r, c = random.choices(choices, weights=weights, k=1)[0]
            frontier.remove((r, c))
            mask[r, c] = True
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != area:
            continue
        tight = _tight_mask(mask)
        if tight.shape[0] < 3 or tight.shape[1] < 3:
            continue
        if bool(tight.all()):
            continue
        if _has_hole(tight):
            continue
        return tight
    return None


def _connected_subset(mask: np.ndarray, size: int) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    start = random.choice(coords)
    chosen = {start}
    frontier = {nbr for nbr in _neighbors_in_mask(start, mask) if nbr not in chosen}
    while len(chosen) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nbr in _neighbors_in_mask(cell, mask):
            if nbr not in chosen:
                frontier.add(nbr)
    out = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        out[r, c] = True
    return out


def _neighbors_in_mask(cell: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _split_fragments(key_mask: np.ndarray, n_parts: int) -> list[np.ndarray] | None:
    area = int(key_mask.sum())
    for _ in range(120):
        parts: list[np.ndarray] = []
        for _part in range(n_parts):
            omit_size = random.randint(max(1, area // 4), max(1, area // 2))
            omit_mask = _connected_subset(key_mask, omit_size)
            part = key_mask & ~omit_mask
            if not part.any() or np.array_equal(part, key_mask):
                break
            parts.append(part)
        if len(parts) != n_parts:
            continue
        union = np.zeros_like(key_mask, dtype=bool)
        for part in parts:
            union |= part
        if np.array_equal(union, key_mask):
            return parts
    return None


def _make_lock_patch(hole_mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = hole_mask.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    patch = np.full((h + top + bottom, w + left + right), color, dtype=int)
    patch[top : top + h, left : left + w][hole_mask] = BLACK
    hole_coords = np.argwhere(hole_mask) + np.array([[top, left]], dtype=int)
    return patch, hole_coords


def _rects_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int,
) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _place_patches(
    width: int,
    max_top_h: int,
    patches: list[np.ndarray],
    *,
    pad: int = 1,
) -> tuple[int, list[tuple[int, int]]] | None:
    min_top_h = max(p.shape[0] for p in patches)
    if min_top_h > max_top_h:
        return None

    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for _ in range(80):
        top_h = random.randint(min_top_h, max_top_h)
        rects: list[tuple[int, int, int, int] | None] = [None] * len(patches)
        positions: list[tuple[int, int] | None] = [None] * len(patches)
        ok = True
        for idx in order:
            ph, pw = patches[idx].shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(top_h - ph + 1):
                for c0 in range(width - pw + 1):
                    rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                    if all(
                        other is None or not _rects_too_close(rect, other, pad)
                        for other in rects
                    ):
                        candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            positions[idx] = (r0, c0)
            rects[idx] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if ok:
            return top_h, [pos for pos in positions if pos is not None]
    return None


def _build_band(piece_masks: list[np.ndarray], fragment_color: int) -> np.ndarray:
    h, w = piece_masks[0].shape
    n = len(piece_masks)
    band = np.zeros((h, n * w + n + 1), dtype=int)
    divider_cols = [i * (w + 1) for i in range(n + 1)]
    band[:, divider_cols] = GREY
    for i, piece in enumerate(piece_masks):
        c0 = divider_cols[i] + 1
        band[:, c0 : c0 + w][piece] = fragment_color
    return band


def _assemble_scene(
    entries: list[dict[str, object]],
    piece_masks: list[np.ndarray],
    fragment_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    band = _build_band(piece_masks, fragment_color)
    gap_rows = random.randint(1, 2)
    max_top_h = 30 - band.shape[0] - gap_rows
    placed = _place_patches(band.shape[1], max_top_h, [entry["patch"] for entry in entries])
    if placed is None:
        return None

    top_h, positions = placed
    grid = np.zeros((top_h + gap_rows + band.shape[0], band.shape[1]), dtype=int)
    target_hole_coords = None

    for entry, (r0, c0) in zip(entries, positions):
        patch = entry["patch"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        if entry["is_target"]:
            target_hole_coords = entry["hole_coords"] + np.array([[r0, c0]], dtype=int)

    strip_top = grid.shape[0] - band.shape[0]
    grid[strip_top:, :] = band

    assert target_hole_coords is not None
    expected = grid.copy()
    expected[target_hole_coords[:, 0], target_hole_coords[:, 1]] = fragment_color
    expected[strip_top:, :] = BLACK
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    fragment_color = random.choice(PALETTE)
    key_mask = _random_connected_mask()
    if key_mask is None:
        return None

    n_parts = random.randint(3, 4)
    piece_masks = _split_fragments(key_mask, n_parts)
    if piece_masks is None:
        return None

    n_objects = 3 if key_mask.shape[1] <= 4 else random.randint(3, 4)
    key_area = int(key_mask.sum())
    same_area_distractor = _random_connected_mask(exact_area=key_area)
    if same_area_distractor is None or _mask_signature(same_area_distractor) == _mask_signature(key_mask):
        return None

    hole_masks = [key_mask, same_area_distractor]
    seen = {_mask_signature(mask) for mask in hole_masks}
    while len(hole_masks) < n_objects:
        exact_area = key_area if random.random() < 0.35 else None
        candidate = _random_connected_mask(exact_area=exact_area)
        if candidate is None:
            return None
        signature = _mask_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        hole_masks.append(candidate)

    hole_signatures = {_mask_signature(mask) for mask in hole_masks}
    if any(_mask_signature(part) in hole_signatures for part in piece_masks):
        return None

    object_colors = random.sample([color for color in PALETTE if color != fragment_color], len(hole_masks))
    entries: list[dict[str, object]] = []
    for idx, (hole_mask, color) in enumerate(zip(hole_masks, object_colors)):
        patch, hole_coords = _make_lock_patch(hole_mask, color)
        entries.append(
            {
                "patch": patch,
                "hole_coords": hole_coords,
                "is_target": idx == 0,
                "hole_mask": hole_mask,
            }
        )
    random.shuffle(entries)

    assembled = _assemble_scene(entries, piece_masks, fragment_color)
    if assembled is None:
        return None
    grid, expected = assembled

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
