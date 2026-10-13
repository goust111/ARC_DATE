import itertools
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
OPS = ("flip_ud", "flip_lr", "rot180")
CARD4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flatten()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in CARD4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _apply_op_mask(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "flip_ud":
        return np.flipud(mask)
    if op == "flip_lr":
        return np.fliplr(mask)
    if op == "rot180":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown op: {op}")


def _transform_panel_mask(panel_mask: np.ndarray, op: str) -> np.ndarray:
    coords = np.argwhere(panel_mask)
    out = np.zeros_like(panel_mask, dtype=bool)
    if coords.size == 0:
        return out
    bbox = bbox_from_coords(coords)
    crop = crop_bbox(panel_mask.astype(int), bbox).astype(bool)
    transformed = _apply_op_mask(crop, op)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _classify_instruction(coords: np.ndarray) -> str:
    bbox = bbox_from_coords(coords)
    crop = _mask_from_coords(coords - np.array([bbox[0], bbox[1]], dtype=int), (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1))
    h, w = crop.shape
    area = int(crop.sum())
    if h == 1 and w >= 2 and area == w:
        return "flip_ud"
    if w == 1 and h >= 2 and area == h:
        return "flip_lr"
    if h == w and area == h * w:
        return "rot180"
    raise AssertionError("instruction piece must be a full horizontal bar, full vertical bar, or filled square")


def _extract_panel_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    n = int(g.shape[0])
    assert g.shape[1] == 3 * n + 2, "expected three square panels"
    assert np.all(g[:, n] == BLACK) and np.all(g[:, 2 * n + 1] == BLACK), "expected black separator columns"

    bg = get_bg(g)
    assert bg != BLACK, "panel background must differ from black separators"
    starts = (0, n + 1, 2 * n + 2)
    records = []

    for start in starts:
        panel = g[:, start : start + n]
        panel_colors = colors_present(panel, exclude={bg})
        assert len(panel_colors) == 1, "each panel must contain exactly one foreground color"
        color = int(panel_colors[0])
        assert set(map(int, np.unique(panel))) <= {bg, color}, "panel must only contain background and one foreground color"

        comps = _find_color_components(panel, color)
        assert len(comps) == 2, "each panel must contain exactly two 4-connected same-color objects"
        comps.sort(key=lambda arr: int(arr.shape[0]))
        small, large = comps
        assert int(small.shape[0]) < int(large.shape[0]), "instruction piece must be strictly smaller than the sprite"

        op = _classify_instruction(small)
        sprite_mask = _mask_from_coords(large, panel.shape)
        records.append({"color": color, "sprite_mask": sprite_mask, "op": op})

    colors = [rec["color"] for rec in records]
    assert len(set(colors)) == 3, "panel colors must be distinct"
    return bg, records


def _overlay_records(records: list[dict], bg: int, ops: tuple[str, str, str], order: tuple[int, int, int] = (0, 1, 2)) -> np.ndarray:
    n = int(records[0]["sprite_mask"].shape[0])
    out = np.full((n, n), int(bg), dtype=int)
    for idx in order:
        rec = records[int(idx)]
        transformed = _transform_panel_mask(rec["sprite_mask"], ops[int(idx)])
        out[transformed] = int(rec["color"])
    return out


def _overlay_raw(records: list[dict], bg: int, order: tuple[int, int, int] = (0, 1, 2)) -> np.ndarray:
    n = int(records[0]["sprite_mask"].shape[0])
    out = np.full((n, n), int(bg), dtype=int)
    for idx in order:
        rec = records[int(idx)]
        out[rec["sprite_mask"]] = int(rec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each panel sprite by its same-colored instruction piece, then overlay the three panels with rightmost priority.

    Input structure:
    - Three equal square panels are arranged left to right and separated by single black columns.
    - Every panel has a uniform non-black background and exactly one foreground color.
    - That foreground color forms exactly two 4-connected components: a larger sprite and a smaller instruction piece shaped as a horizontal bar, vertical bar, or filled square.

    Rule:
    - In each panel, read the smaller component as the instruction: a horizontal bar flips the sprite top-to-bottom, a vertical bar flips it left-to-right, and a filled square rotates it by 180 degrees.
    - Apply the transform only within the large sprite's own bounding box, keeping the transformed sprite anchored at the same panel coordinates.
    - Remove the instruction pieces and overlay the three transformed sprites onto one panel-sized output, with later panels covering earlier ones on overlaps.

    Key insight:
    - The tiny same-colored component is not noise; its shape encodes the geometric transform that must be applied before the panels are merged.
    """
    bg, records = _extract_panel_records(grid)
    ops = tuple(str(rec["op"]) for rec in records)
    return _overlay_records(records, bg, ops)


def _random_connected_crop(panel_size: int, rng: random.Random) -> np.ndarray | None:
    max_side = min(5, panel_size - 1)
    for _ in range(120):
        h = rng.randint(2, max_side)
        w = rng.randint(2, max_side)
        max_area = min(h * w - 1, 9)
        min_area = min(max_area, 5)
        if max_area < min_area:
            continue
        area = rng.randint(min_area, max_area)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in CARD4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        bbox = bbox_from_coords(coords)
        crop = crop_bbox(mask.astype(int), bbox).astype(bool)
        if crop.shape[0] < 2 or crop.shape[1] < 2 or int(crop.sum()) < 5:
            continue
        variants = [crop, _apply_op_mask(crop, "flip_ud"), _apply_op_mask(crop, "flip_lr"), _apply_op_mask(crop, "rot180")]
        if len({variant.tobytes() for variant in variants}) != 4:
            continue
        return crop
    return None


def _panel_from_crop(crop: np.ndarray, top: int, left: int, panel_size: int) -> np.ndarray:
    h, w = crop.shape
    out = np.zeros((panel_size, panel_size), dtype=bool)
    out[top : top + h, left : left + w] = crop
    return out


def _all_placements(crop: np.ndarray, panel_size: int) -> list[np.ndarray]:
    h, w = crop.shape
    out = []
    for top in range(panel_size - h + 1):
        for left in range(panel_size - w + 1):
            out.append(_panel_from_crop(crop, top, left, panel_size))
    return out


def _overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a & b))


def _instruction_crop(op: str, sprite_area: int, rng: random.Random) -> np.ndarray:
    if op == "flip_ud":
        return np.ones((1, rng.choice([2, 3])), dtype=bool)
    if op == "flip_lr":
        return np.ones((rng.choice([2, 3]), 1), dtype=bool)
    side = 2 if sprite_area >= 7 and rng.random() < 0.5 else 1
    return np.ones((side, side), dtype=bool)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r0, r1 = max(0, int(r) - 1), min(h, int(r) + 2)
        c0, c1 = max(0, int(c) - 1), min(w, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _place_instruction(sprite_mask: np.ndarray, op: str, rng: random.Random) -> np.ndarray | None:
    panel_size = int(sprite_mask.shape[0])
    crop = _instruction_crop(op, int(sprite_mask.sum()), rng)
    h, w = crop.shape
    candidates = []
    sprite_halo = _dilate8(sprite_mask)
    for top in range(panel_size - h + 1):
        for left in range(panel_size - w + 1):
            candidate = _panel_from_crop(crop, top, left, panel_size)
            if np.any(candidate & sprite_halo):
                continue
            candidates.append(candidate)
    if not candidates:
        return None
    return rng.choice(candidates)


def _place_final_sprites(crops: list[np.ndarray], panel_size: int, rng: random.Random) -> list[np.ndarray] | None:
    branch = rng.choice(["mid_over_left", "right_over_mid", "right_over_both"])
    placements = [_all_placements(crop, panel_size) for crop in crops]

    for _ in range(150):
        mask0 = rng.choice(placements[0])

        if branch == "mid_over_left":
            cand1 = [
                m1
                for m1 in placements[1]
                if 1 <= _overlap(m1, mask0) <= min(3, int(mask0.sum()) - 1, int(m1.sum()) - 1)
            ]
            if not cand1:
                continue
            mask1 = rng.choice(cand1)
            union01 = mask0 | mask1
            cand2 = [m2 for m2 in placements[2] if _overlap(m2, union01) == 0]
            if not cand2:
                continue
            mask2 = rng.choice(cand2)

        elif branch == "right_over_mid":
            cand1 = [m1 for m1 in placements[1] if _overlap(m1, mask0) == 0]
            if not cand1:
                continue
            mask1 = rng.choice(cand1)
            cand2 = [
                m2
                for m2 in placements[2]
                if _overlap(m2, mask0) == 0
                and 1 <= _overlap(m2, mask1) <= min(3, int(mask1.sum()) - 1, int(m2.sum()) - 1)
            ]
            if not cand2:
                continue
            mask2 = rng.choice(cand2)

        else:
            cand1 = [m1 for m1 in placements[1] if _overlap(m1, mask0) == 0]
            if not cand1:
                continue
            mask1 = rng.choice(cand1)
            cand2 = []
            for m2 in placements[2]:
                ov0 = _overlap(m2, mask0)
                ov1 = _overlap(m2, mask1)
                total = _overlap(m2, mask0 | mask1)
                if ov0 >= 1 and ov1 >= 1 and total <= 4:
                    cand2.append(m2)
            if not cand2:
                continue
            mask2 = rng.choice(cand2)

        visible0 = int(np.count_nonzero(mask0 & ~mask1 & ~mask2))
        visible1 = int(np.count_nonzero(mask1 & ~mask2))
        visible2 = int(mask2.sum())
        total_overlap = _overlap(mask1, mask0) + _overlap(mask2, mask0 | mask1)
        if visible0 > 0 and visible1 > 0 and visible2 > 0 and 1 <= total_overlap <= 4:
            return [mask0, mask1, mask2]
    return None


def _compose_input(panel_size: int, bg: int, colors: list[int], final_masks: list[np.ndarray], ops: list[str], rng: random.Random) -> np.ndarray | None:
    panels = []
    for color, final_mask, op in zip(colors, final_masks, ops):
        input_sprite = _transform_panel_mask(final_mask, op)
        instruction = _place_instruction(input_sprite, op, rng)
        if instruction is None:
            return None
        panel = np.full((panel_size, panel_size), int(bg), dtype=int)
        panel[input_sprite] = int(color)
        panel[instruction] = int(color)
        panels.append(panel)

    grid = np.full((panel_size, 3 * panel_size + 2), BLACK, dtype=int)
    starts = (0, panel_size + 1, 2 * panel_size + 2)
    for start, panel in zip(starts, panels):
        grid[:, start : start + panel_size] = panel
    return grid


def _expected_output(panel_size: int, bg: int, colors: list[int], final_masks: list[np.ndarray]) -> np.ndarray:
    out = np.full((panel_size, panel_size), int(bg), dtype=int)
    for color, mask in zip(colors, final_masks):
        out[mask] = int(color)
    return out


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray, ops: list[str]) -> bool:
    try:
        produced = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(produced, expected):
        return False

    bg, records = _extract_panel_records(inp)
    intended = tuple(ops)
    if not np.array_equal(_overlay_records(records, bg, intended), expected):
        return False
    if np.array_equal(_overlay_raw(records, bg), expected):
        return False
    if np.array_equal(_overlay_records(records, bg, intended, order=(2, 1, 0)), expected):
        return False

    hits = []
    for combo in itertools.product(OPS, repeat=3):
        if np.array_equal(_overlay_records(records, bg, combo), expected):
            hits.append(combo)
    if hits != [intended]:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_size = rng.randint(6, 9)
    palette = list(range(1, 10))
    bg = rng.choice(palette)
    palette.remove(bg)
    colors = rng.sample(palette, 3)
    ops = list(OPS)
    rng.shuffle(ops)

    crops = []
    for _ in range(3):
        crop = _random_connected_crop(panel_size, rng)
        if crop is None:
            return None
        crops.append(crop)

    final_masks = _place_final_sprites(crops, panel_size, rng)
    if final_masks is None:
        return None

    inp = _compose_input(panel_size, bg, colors, final_masks, ops, rng)
    if inp is None:
        return None
    expected = _expected_output(panel_size, bg, colors, final_masks)
    if not _validate_generated_case(inp, expected, ops):
        return None

    sig = (tuple(expected.shape), expected.tobytes())
    if sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt(random)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid panel-instruction example")
