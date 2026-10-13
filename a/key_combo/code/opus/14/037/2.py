from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
INK = 5
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _symmetry_type(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    if horizontal and not vertical:
        return "H"
    if vertical and not horizontal:
        return "V"
    return "N"


def _edge_counts(mask: np.ndarray) -> tuple[int, int, int, int]:
    return int(mask[0].sum()), int(mask[-1].sum()), int(mask[:, 0].sum()), int(mask[:, -1].sum())


def _row_pref(mask: np.ndarray) -> bool:
    top = tuple(map(int, mask[0]))
    rev_top = tuple(map(int, mask[0, ::-1]))
    assert top != rev_top, "mask must expose a horizontal orientation preference"
    return top > rev_top


def _col_pref(mask: np.ndarray) -> bool:
    left = tuple(map(int, mask[:, 0]))
    rev_left = tuple(map(int, mask[::-1, 0]))
    assert left != rev_left, "mask must expose a vertical orientation preference"
    return left > rev_left


def _build_slot_from_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = mask.shape
    kind = _symmetry_type(mask)
    top, bottom, left, right = _edge_counts(mask)

    if kind == "H":
        left_oriented = _row_pref(mask)
        slot = np.zeros((h + 2, w + 1), dtype=bool)
        slot[0, :] = True
        slot[-1, :] = True
        if left_oriented:
            slot[1:-1, 0] = True
            return slot, (1, 1)
        slot[1:-1, -1] = True
        return slot, (1, 0)

    if kind == "V":
        top_oriented = _col_pref(mask)
        slot = np.zeros((h + 1, w + 2), dtype=bool)
        slot[:, 0] = True
        slot[:, -1] = True
        if top_oriented:
            slot[0, 1:-1] = True
            return slot, (1, 1)
        slot[-1, 1:-1] = True
        return slot, (0, 1)

    assert top != bottom and left != right, "asymmetric sprite must have both directional biases"
    slot = np.zeros((h + 2, w + 2), dtype=bool)
    top_heavy = top > bottom
    left_heavy = left > right
    if top_heavy:
        slot[0, :] = True
        partial = slice(0, h + 1)
        cavity_r = 1
    else:
        slot[-1, :] = True
        partial = slice(1, h + 2)
        cavity_r = 0
    if left_heavy:
        slot[:, 0] = True
        slot[partial, -1] = True
        cavity_c = 1
    else:
        slot[:, -1] = True
        slot[partial, 0] = True
        cavity_c = 0
    return slot, (cavity_r, cavity_c)


def _transforms(arr: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        rot = np.rot90(arr, k)
        out.append(rot)
        out.append(np.fliplr(rot))
    uniq = []
    seen = set()
    for candidate in out:
        key = (candidate.shape, candidate.tobytes())
        if key not in seen:
            seen.add(key)
            uniq.append(candidate.copy())
    return uniq


def _apply_transform(arr: np.ndarray, op: int) -> np.ndarray:
    rot = np.rot90(arr, op % 4)
    return np.fliplr(rot) if op >= 4 else rot


def _paired_transforms(full_mask: np.ndarray, grey_mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    seen = set()
    for op in range(8):
        full_t = _apply_transform(full_mask, op)
        grey_t = _apply_transform(grey_mask, op)
        key = (full_t.shape, full_t.tobytes(), grey_t.tobytes())
        if key not in seen:
            seen.add(key)
            pairs.append((full_t.copy(), grey_t.copy()))
    return pairs


def _parse_upper(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BG
        colors = sorted(set(map(int, patch[mask])))
        assert INK in colors, "each upper sprite must contain grey ink"
        signature = [c for c in colors if c != INK]
        assert len(signature) == 1, "each upper sprite must have exactly one signature color"
        grey_mask = patch == INK
        full_mask = mask
        kind = _symmetry_type(full_mask)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch.copy(),
                "full_mask": full_mask.copy(),
                "grey_mask": grey_mask.copy(),
                "kind": kind,
                "signature_color": signature[0],
            }
        )
    assert len(objects) == 3, "expected exactly three legend sprites"
    kinds = {obj["kind"] for obj in objects}
    assert kinds == {"H", "V", "N"}, "legend sprites must realize the three symmetry classes"
    return objects


def _parse_lower(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, patch[patch != BG])))
        assert len(colors) == 1, "each lower slot must be monochrome"
        slot_mask = patch != BG
        kind = _symmetry_type(slot_mask)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "slot_mask": slot_mask.copy(),
                "color": colors[0],
                "kind": kind,
            }
        )
    assert len(objects) == 3, "expected exactly three target slots"
    kinds = {obj["kind"] for obj in objects}
    assert kinds == {"H", "V", "N"}, "target slots must realize the three symmetry classes"
    return objects


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    fg_rows = np.flatnonzero((grid != BG).any(axis=1))
    assert len(fg_rows) > 0, "grid must contain foreground"
    gaps = np.diff(fg_rows)
    assert np.any(gaps > 1), "expected a blank separator row"
    cut_idx = int(np.argmax(gaps))
    top_end = int(fg_rows[cut_idx])
    bottom_start = int(fg_rows[cut_idx + 1])
    sep_row = top_end + 1
    assert sep_row < bottom_start and np.all(grid[sep_row:bottom_start] == BG), "separator must be blank"
    return grid[:top_end + 1, :], grid[bottom_start:, :], bottom_start


def _matched_transform(sprite: dict, slot: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    candidates = []
    for full_mask, grey_mask in _paired_transforms(sprite["full_mask"], sprite["grey_mask"]):
        if not np.all(grey_mask <= full_mask):
            continue
        expected_slot, cavity = _build_slot_from_mask(full_mask)
        if np.array_equal(expected_slot, slot["slot_mask"]):
            candidates.append((full_mask.copy(), grey_mask.copy(), cavity))
    uniq = []
    seen = set()
    for full_mask, grey_mask, cavity in candidates:
        key = (full_mask.shape, full_mask.tobytes(), grey_mask.tobytes(), cavity)
        if key not in seen:
            seen.add(key)
            uniq.append((full_mask, grey_mask, cavity))
    assert len(uniq) == 1, "each sprite-slot pair must admit exactly one aligned transform"
    return uniq[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each legend sprite to the target slot with the same reflection-symmetry class, then stamp only the grey ink into that slot using the slot's color.

    Input structure:
    - The grid has a black background and is split into an upper legend region and a lower work region by a blank black row.
    - The upper region contains three separated 4-connected multicolor sprites made of grey ink cells plus one distinct signature color per sprite.
    - The lower region contains three separated monochrome hollow slot outlines whose symmetry classes are respectively horizontal-only, vertical-only, and none.

    Rule:
    - Classify each upper sprite and lower slot by reflection symmetry of its occupied-cell mask: horizontal-only, vertical-only, or none (including both-or-neither).
    - Pair each sprite with the unique slot of the same symmetry class.
    - For each pair, rotate and/or reflect the sprite until its full occupied mask aligns with the slot outline's template shape, then stamp only the sprite's grey cells into the slot cavity and recolor them to the slot outline color.
    - Do not stamp the signature-color cells, keep the slot outline itself, and erase all original upper sprites to black.

    Key insight:
    - Symmetry is not the output label here; it is the matching key that tells you which sprite belongs to which slot, while the slot outline's geometry fixes the needed orientation before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(_get_bg(g)) == BG, "background must be black"

    top_region, bottom_region, bottom_offset = _split_regions(g)
    upper = _parse_upper(top_region)
    lower = _parse_lower(bottom_region)
    slot_by_kind = {slot["kind"]: slot for slot in lower}

    out = g.copy()
    out[:bottom_offset, :] = BG

    for sprite in upper:
        slot = slot_by_kind[sprite["kind"]]
        _full_mask, grey_mask, cavity = _matched_transform(sprite, slot)
        r0, c0, _r1, _c1 = slot["bbox"]
        cr, cc = cavity
        rr = bottom_offset + r0 + cr
        cc_abs = c0 + cc
        h, w = grey_mask.shape
        patch = out[rr : rr + h, cc_abs : cc_abs + w]
        patch[grey_mask] = int(slot["color"])

    return out


SPRITE_LIBRARY = {
    "H": [
        np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    ],
    "V": [
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool),
    ],
    "N": [
        np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    ],
}


def _random_sprite_mask(kind: str) -> np.ndarray:
    candidates = []
    for base in SPRITE_LIBRARY[kind]:
        for transformed in _transforms(base):
            if _symmetry_type(transformed) != kind:
                continue
            try:
                slot, _ = _build_slot_from_mask(transformed)
            except AssertionError:
                continue
            if 8 <= int(slot.sum()) <= 12:
                candidates.append(transformed.copy())
    assert candidates, f"no usable masks for {kind}"
    return random.choice(candidates)


def _paint_sprite(mask: np.ndarray, signature_color: int) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    sig_count = 1 if len(coords) <= 5 or random.random() < 0.5 else 2
    sig_cells = set(random.sample(coords, sig_count))
    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[r, c] = signature_color if (r, c) in sig_cells else INK
    assert np.any(sprite == INK) and np.any(sprite == signature_color)
    return sprite


def _fits(scene: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = scene.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    return not np.any(scene[rr0:rr1, cc0:cc1] != BG)


def _paste(scene: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    scene[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]


def _sample_positions(widths: list[int], region_w: int, margin: int = 1) -> list[int]:
    total = sum(widths)
    min_gap_sum = 2 * (len(widths) - 1)
    slack = region_w - 2 * margin - total - min_gap_sum
    if slack < 0:
        raise ValueError("region too narrow")
    bins = [0] * (len(widths) + 1)
    for _ in range(slack):
        bins[random.randrange(len(bins))] += 1
    xs = [margin + bins[0]]
    for i in range(1, len(widths)):
        gap = 2 + bins[i]
        xs.append(xs[-1] + widths[i - 1] + gap)
    return xs


def _build_example() -> np.ndarray:
    signature_palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 6)
    signature_colors = signature_palette[:3]
    slot_colors = signature_palette[3:]

    sprites = []
    slots = []
    for kind, sig_color, slot_color in zip(random.sample(["H", "V", "N"], 3), signature_colors, slot_colors):
        mask = _random_sprite_mask(kind)
        sprite = _paint_sprite(mask, sig_color)
        transformed_masks = [m for m in _transforms(mask) if _symmetry_type(m) == kind]
        random.shuffle(transformed_masks)
        chosen = transformed_masks[0]
        slot_mask, _ = _build_slot_from_mask(chosen)
        slot_patch = np.where(slot_mask, slot_color, BG)
        sprites.append({"kind": kind, "patch": sprite, "full_mask": mask})
        slots.append({"kind": kind, "patch": slot_patch, "slot_mask": slot_mask})

    random.shuffle(sprites)
    random.shuffle(slots)

    top_h = max(p["patch"].shape[0] for p in sprites) + random.randint(2, 4)
    bottom_h = max(p["patch"].shape[0] for p in slots) + random.randint(2, 4)
    widths = [max(s["patch"].shape[1], t["patch"].shape[1]) for s, t in zip(sprites, slots)]
    width = max(18, min(30, sum(widths) + 2 * (len(widths) - 1) + random.randint(5, 9)))

    top_cols = _sample_positions([obj["patch"].shape[1] for obj in sprites], width)
    bottom_cols = _sample_positions([obj["patch"].shape[1] for obj in slots], width)
    grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)

    top_band = random.randint(0, max(0, top_h - max(obj["patch"].shape[0] for obj in sprites)))
    for obj, c0 in zip(sprites, top_cols):
        patch = obj["patch"]
        r0 = top_band
        assert _fits(grid[:top_h, :], patch, r0, c0, pad=1)
        _paste(grid[:top_h, :], patch, r0, c0)

    bottom_region = grid[top_h + 1 :, :]
    bottom_band = random.randint(0, max(0, bottom_h - max(obj["patch"].shape[0] for obj in slots)))
    for obj, c0 in zip(slots, bottom_cols):
        patch = obj["patch"]
        r0 = bottom_band
        assert _fits(bottom_region, patch, r0, c0, pad=1)
        _paste(bottom_region, patch, r0, c0)

    return grid


def _check_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    top_region, bottom_region, bottom_offset = _split_regions(inp)
    upper = _parse_upper(top_region)
    lower = _parse_lower(bottom_region)
    assert np.all(out[:bottom_offset, :] == BG), "upper sprites must be erased"

    lower_out = out[bottom_offset:, :]
    for slot in lower:
        slot_pixels = lower_out[slot["bbox"][0] : slot["bbox"][2] + 1, slot["bbox"][1] : slot["bbox"][3] + 1]
        assert np.all(slot_pixels[slot["slot_mask"]] == slot["color"]), "slot outline color must remain unchanged"

    slot_by_kind = {slot["kind"]: slot for slot in lower}
    for sprite in upper:
        slot = slot_by_kind[sprite["kind"]]
        full_mask, grey_mask, cavity = _matched_transform(sprite, slot)
        r0, c0, _r1, _c1 = slot["bbox"]
        cr, cc = cavity
        rr = bottom_offset + r0 + cr
        cc_abs = c0 + cc
        stamped = out[rr : rr + grey_mask.shape[0], cc_abs : cc_abs + grey_mask.shape[1]]
        assert np.all(stamped[grey_mask] == slot["color"])
        assert np.all(stamped[full_mask & ~grey_mask] == BG), "signature cells must stay black"

    all_matches = 0
    for slot in lower:
        for sprite in upper:
            if sprite["kind"] != slot["kind"]:
                continue
            _full_mask, grey_mask, _ = _matched_transform(sprite, slot)
            all_matches += int(grey_mask.sum() > 0)
    assert all_matches == 3, "expected one valid match per slot"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp = _build_example()
            out = main(inp)
            _check_semantics(inp, out)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid symmetry-slot ARC example")
