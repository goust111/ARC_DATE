from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8)

ORTHO4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

GRID_W_RANGE = (18, 30)
TOP_H_RANGE = (4, 8)
BOTTOM_H_RANGE = (10, 18)
LEGEND_COUNT_RANGE = (2, 5)
OBJECT_EXTRA_CHOICES = (0, 1, 2)
CORE_H_RANGE = (5, 7)
CORE_W_RANGE = (5, 8)
TAB_THICKNESS_RANGE = (1, 2)
PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> int:
    key = 0
    for r in range(3):
        for c in range(3):
            if bool(mask[r, c]):
                key |= 1 << (r * 3 + c)
    return int(key)


def _key_to_mask(key: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    for idx in range(9):
        if (int(key) >> idx) & 1:
            mask[idx // 3, idx % 3] = True
    return mask


def _connected(mask: np.ndarray, connectivity: int) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    neighbors = ORTHO4 if int(connectivity) == 4 else NEIGHBORS8
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


MASK_LIBRARY = [
    key
    for key in range(1 << 9)
    if 5 <= int(bin(key).count("1")) <= 7
    and _touches_all_sides(_key_to_mask(key))
    and _connected(_key_to_mask(key), connectivity=4)
]


def _components(mask: np.ndarray, connectivity: int) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    neighbors = ORTHO4 if int(connectivity) == 4 else NEIGHBORS8
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in neighbors:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = map(int, arr.min(axis=0))
            r1, c1 = map(int, arr.max(axis=0))
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _separator_row(g: np.ndarray) -> int:
    rows = [
        int(r)
        for r in range(g.shape[0])
        if int(g[r, 0]) != BG and bool(np.all(g[r, :] == g[r, 0]))
    ]
    assert len(rows) == 1, "expected exactly one full non-background separator row"
    return rows[0]


def _dominant_color(patch: np.ndarray) -> int:
    colors = [int(v) for v in patch[patch != BG].tolist()]
    counts = Counter(colors)
    color, freq = max(counts.items(), key=lambda item: item[1])
    assert sum(int(v == freq) for v in counts.values()) == 1, "legend dominant color must be unique"
    return int(color)


def _legend_mapping(top: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in _components(top != BG, connectivity=8):
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "legend sprite must fit a 3x3 window"
        patch = top[r0 : r1 + 1, c0 : c1 + 1]
        key = _mask_key(patch != BG)
        assert key not in mapping, "legend fingerprints must be unique"
        mapping[key] = _dominant_color(patch)
    assert mapping, "expected at least one legend sprite"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lower object's internal maroon fingerprint to an upper 3x3 legend
    sprite and recolor the object's grey body with that legend's dominant color.
    The maroon fingerprint pixels are then cleared to background.
    """
    g = np.asarray(grid, dtype=int)
    sep_r = _separator_row(g)
    out = g.copy()

    mapping = _legend_mapping(g[:sep_r, :])
    lower = g[sep_r + 1 :, :]

    for comp in _components(lower != BG, connectivity=4):
        r0, c0, r1, c1 = comp["bbox"]
        patch = lower[r0 : r1 + 1, c0 : c1 + 1]
        maroon = patch == MAROON
        assert maroon.any(), "each lower object must contain a maroon fingerprint"
        coords = np.argwhere(maroon)
        mr0, mc0 = map(int, coords.min(axis=0))
        mr1, mc1 = map(int, coords.max(axis=0))
        assert mr1 - mr0 + 1 == 3 and mc1 - mc0 + 1 == 3, "fingerprint must occupy a 3x3 window"
        key = _mask_key(maroon[mr0 : mr1 + 1, mc0 : mc1 + 1])
        assert key in mapping, "unknown fingerprint pattern"
        new_color = int(mapping[key])

        out_patch = out[sep_r + 1 + r0 : sep_r + 1 + r1 + 1, c0 : c1 + 1]
        out_patch[out_patch == GREY] = new_color
        out_patch[out_patch == MAROON] = BG

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _legend_patch(shape_key: int, dominant: int) -> np.ndarray:
    mask = _key_to_mask(shape_key)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    area = len(coords)
    dominant_count = random.randint(area // 2 + 1, area - 1)
    remaining = area - dominant_count
    accent_count = 1 if remaining == 1 or random.random() < 0.6 else 2
    accents = random.sample([c for c in LEGEND_COLORS if c != dominant], accent_count)

    colors = [dominant] * dominant_count
    if accent_count == 1:
        colors.extend([accents[0]] * remaining)
    else:
        split = random.randint(1, remaining - 1)
        colors.extend([accents[0]] * split)
        colors.extend([accents[1]] * (remaining - split))
    random.shuffle(colors)

    patch = np.full((3, 3), BG, dtype=int)
    for (r, c), color in zip(coords, colors, strict=False):
        patch[r, c] = int(color)
    return patch


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _place_patches(height: int, width: int, patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda i: int((patches[i] != BG).sum()), reverse=True)

    for idx in order:
        patch = patches[idx]
        mask = patch != BG
        ph, pw = patch.shape
        if ph > height or pw > width:
            return None
        dilated = _dilate8(occupied)
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            if bool((dilated[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                continue
            _paint_patch(canvas, patch, r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            break
        else:
            return None

    return canvas


def _tab_coords(core_h: int, core_w: int) -> set[tuple[int, int]]:
    coords = {(r, c) for r in range(core_h) for c in range(core_w)}
    tab_sides = random.sample(("top", "bottom", "left", "right"), random.randint(0, 2))
    for side in tab_sides:
        thickness = random.randint(*TAB_THICKNESS_RANGE)
        if side in {"top", "bottom"}:
            span = random.randint(2, core_w - 1)
            c0 = random.randint(0, core_w - span)
            rows = range(-thickness, 0) if side == "top" else range(core_h, core_h + thickness)
            for r in rows:
                for c in range(c0, c0 + span):
                    coords.add((int(r), int(c)))
        else:
            span = random.randint(2, core_h - 1)
            r0 = random.randint(0, core_h - span)
            cols = range(-thickness, 0) if side == "left" else range(core_w, core_w + thickness)
            for r in range(r0, r0 + span):
                for c in cols:
                    coords.add((int(r), int(c)))
    return coords


def _object_pair(shape_key: int, target_color: int) -> tuple[np.ndarray, np.ndarray]:
    mask3 = _key_to_mask(shape_key)
    core_h = random.randint(*CORE_H_RANGE)
    core_w = random.randint(*CORE_W_RANGE)
    coords = _tab_coords(core_h, core_w)

    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = {(int(r - min_r), int(c - min_c)) for r, c in coords}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    obj_mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        obj_mask[r, c] = True

    core_r0 = int(-min_r)
    core_c0 = int(-min_c)
    finger_r = core_r0 + random.randint(1, core_h - 4)
    finger_c = core_c0 + random.randint(1, core_w - 4)

    inp = np.full((h, w), BG, dtype=int)
    inp[obj_mask] = GREY
    in_window = inp[finger_r : finger_r + 3, finger_c : finger_c + 3]
    in_window[mask3] = MAROON

    out = np.full((h, w), BG, dtype=int)
    out[obj_mask] = target_color
    out_window = out[finger_r : finger_r + 3, finger_c : finger_c + 3]
    out_window[mask3] = BG
    return inp, out


def _place_object_pairs(height: int, width: int, pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = list(range(len(pairs)))
    random.shuffle(order)
    order.sort(key=lambda i: int((pairs[i][0] != BG).sum()), reverse=True)

    for idx in order:
        inp_patch, out_patch = pairs[idx]
        mask = inp_patch != BG
        ph, pw = inp_patch.shape
        if ph > height or pw > width:
            return None
        dilated = _dilate8(occupied)
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            if bool((dilated[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                continue
            _paint_patch(inp, inp_patch, r0, c0)
            _paint_patch(out, out_patch, r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            break
        else:
            return None

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GRID_W_RANGE)
    top_h = random.randint(*TOP_H_RANGE)
    bottom_h = random.randint(*BOTTOM_H_RANGE)
    legend_count = random.randint(*LEGEND_COUNT_RANGE)

    if len(MASK_LIBRARY) < legend_count:
        return None

    legend_keys = random.sample(MASK_LIBRARY, legend_count)
    dominant_colors = random.sample(LEGEND_COLORS, legend_count)
    legend_patches = [
        _legend_patch(shape_key, color)
        for shape_key, color in zip(legend_keys, dominant_colors, strict=False)
    ]
    top = _place_patches(top_h, width, legend_patches)
    if top is None:
        return None

    object_count = legend_count + random.choice(OBJECT_EXTRA_CHOICES)
    assignments = list(range(legend_count))
    assignments.extend(random.randrange(legend_count) for _ in range(object_count - legend_count))
    random.shuffle(assignments)

    object_pairs = [
        _object_pair(legend_keys[idx], dominant_colors[idx])
        for idx in assignments
    ]
    placed = _place_object_pairs(bottom_h, width, object_pairs)
    if placed is None:
        return None
    bottom_inp, bottom_out = placed

    inp = np.full((top_h + 1 + bottom_h, width), BG, dtype=int)
    out = np.full_like(inp, BG)
    inp[:top_h, :] = top
    out[:top_h, :] = top
    inp[top_h, :] = GREY
    out[top_h, :] = GREY
    inp[top_h + 1 :, :] = bottom_inp
    out[top_h + 1 :, :] = bottom_out

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
