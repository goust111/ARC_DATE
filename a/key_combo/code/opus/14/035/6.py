from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
ORANGE = 7

GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    cur = mask
    for _ in range(4):
        variants.append(cur)
        variants.append(np.fliplr(cur))
        cur = np.rot90(cur)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for variant in variants:
        key = (variant.shape, variant.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(variant)
    return out


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _d4_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    return any(_same_mask(variant, b) for variant in _mask_variants(a))


def _is_asymmetric(mask: np.ndarray) -> bool:
    return all(not _same_mask(mask, variant) for variant in _mask_variants(mask)[1:])


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.where(mask)
    grid[r0 + rr, c0 + cc] = int(color)


def _draw_hollow_box(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0:c0 + w] = GRAY
    grid[r0 + h - 1, c0:c0 + w] = GRAY
    grid[r0:r0 + h, c0] = GRAY
    grid[r0:r0 + h, c0 + w - 1] = GRAY


def _is_hollow_box(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _find_reference_records(grid: np.ndarray) -> list[dict]:
    gray_comps = _components(grid == GRAY, connectivity=4)
    color_comps = _components((grid != BG) & (grid != GRAY), connectivity=4)

    references: list[dict] = []
    for gray_coords in gray_comps:
        if not _is_hollow_box(grid, gray_coords):
            continue
        r0, c0, r1, c1 = _bbox(gray_coords)
        candidates = []
        for coords in color_comps:
            cr0, cc0, cr1, cc1 = _bbox(coords)
            if r0 < cr0 <= cr1 < r1 and c0 < cc0 <= cc1 < c1:
                candidates.append(coords)
        if len(candidates) != 1:
            continue
        ref_coords = candidates[0]
        ref_mask = _mask_from_coords(ref_coords)
        ref_color = int(grid[ref_coords[0, 0], ref_coords[0, 1]])
        references.append(
            {
                "box_bbox": (r0, c0, r1, c1),
                "coords": ref_coords,
                "mask": ref_mask,
                "color": ref_color,
            }
        )
    return references


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each free orange sprite to the opposite prototype, then recolor it from the matching boxed reference.

    Input structure:
    - The background is black.
    - Two gray hollow rectangles each contain one small colored reference sprite.
    - Several free orange sprites outside the boxes are exact translated copies of one reference shape or the other.

    Rule:
    - Detect the two gray boxes and extract the reference sprite inside each box, recording its shape mask and color.
    - For every free orange sprite, match its current shape to one reference mask.
    - Erase that orange sprite, redraw the opposite reference mask with its top-left aligned to the old sprite bbox top-left, and color it with the opposite reference's color.
    - Leave the gray boxes and the boxed reference sprites unchanged.

    Key insight:
    - Color transfer is driven by the sprite's shape after the prototype swap, so original A-shaped sprites end as B-shaped and take B's reference color, not A's.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    references = _find_reference_records(g)
    assert len(references) == 2, "expected exactly two boxed references"
    assert not _same_mask(references[0]["mask"], references[1]["mask"]), "reference masks must differ"

    free_orange = _components(g == ORANGE, connectivity=4)
    assert free_orange, "expected free orange sprites"

    for coords in free_orange:
        sprite_mask = _mask_from_coords(coords)
        matches = [i for i, ref in enumerate(references) if _same_mask(sprite_mask, ref["mask"])]
        assert len(matches) == 1, "each orange sprite must match exactly one reference shape"
        src_idx = matches[0]
        dst_idx = 1 - src_idx
        r0, c0, _, _ = _bbox(coords)
        out[coords[:, 0], coords[:, 1]] = BG
        _paste_mask(out, references[dst_idx]["mask"], r0, c0, references[dst_idx]["color"])

    return out


def _random_shape_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(5, h * w))
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        cells = [start]
        while int(mask.sum()) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
            cells.append((nr, nc))
        coords = np.argwhere(mask)
        if len(coords) != area:
            continue
        cropped = _mask_from_coords(coords)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if _is_asymmetric(cropped):
            return cropped
    raise RuntimeError("failed to sample asymmetric connected mask")


def _sample_reference_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        a = _random_shape_mask()
        b = _random_shape_mask()
        if _same_mask(a, b):
            continue
        if _d4_equivalent(a, b):
            continue
        if abs(int(a.sum()) - int(b.sum())) + (a.shape != b.shape) < 1:
            continue
        return a, b
    raise RuntimeError("failed to sample non-equivalent reference pair")


def _all_positions(h: int, w: int, obj_h: int, obj_w: int, margin: int = 1) -> list[tuple[int, int]]:
    positions = []
    for r in range(margin, h - obj_h - margin + 1):
        for c in range(margin, w - obj_w - margin + 1):
            positions.append((r, c))
    random.shuffle(positions)
    return positions


def _can_place_rect(occ: np.ndarray, r: int, c: int, h: int, w: int, gap: int = 1) -> bool:
    h0, w0 = occ.shape
    r0 = max(0, r - gap)
    c0 = max(0, c - gap)
    r1 = min(h0, r + h + gap)
    c1 = min(w0, c + w + gap)
    return not occ[r0:r1, c0:c1].any()


def _mark_rect(occ: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occ[r:r + h, c:c + w] = True


def _build_expected(
    inp: np.ndarray,
    free_specs: list[dict],
    proto_a: np.ndarray,
    proto_b: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    for spec in free_specs:
        r0 = spec["r"]
        c0 = spec["c"]
        old = proto_a if spec["kind"] == "A" else proto_b
        new = proto_b if spec["kind"] == "A" else proto_a
        color = BLUE if spec["kind"] == "A" else RED
        rr, cc = np.where(old)
        out[r0 + rr, c0 + cc] = BG
        _paste_mask(out, new, r0, c0, color)
    return out


def _build_wrong_after_swap_original_color(
    inp: np.ndarray,
    free_specs: list[dict],
    proto_a: np.ndarray,
    proto_b: np.ndarray,
) -> np.ndarray:
    wrong = inp.copy()
    for spec in free_specs:
        r0 = spec["r"]
        c0 = spec["c"]
        old = proto_a if spec["kind"] == "A" else proto_b
        new = proto_b if spec["kind"] == "A" else proto_a
        color = RED if spec["kind"] == "A" else BLUE
        rr, cc = np.where(old)
        wrong[r0 + rr, c0 + cc] = BG
        _paste_mask(wrong, new, r0, c0, color)
    return wrong


def _build_wrong_no_swap(
    inp: np.ndarray,
    free_specs: list[dict],
    proto_a: np.ndarray,
    proto_b: np.ndarray,
) -> np.ndarray:
    wrong = inp.copy()
    for spec in free_specs:
        r0 = spec["r"]
        c0 = spec["c"]
        old = proto_a if spec["kind"] == "A" else proto_b
        rr, cc = np.where(old)
        wrong[r0 + rr, c0 + cc] = RED if spec["kind"] == "A" else BLUE
    return wrong


def _generate_case() -> np.ndarray:
    h = random.randint(19, 26)
    w = random.randint(19, 26)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    proto_a, proto_b = _sample_reference_pair()
    max_h = max(proto_a.shape[0], proto_b.shape[0])
    max_w = max(proto_a.shape[1], proto_b.shape[1])

    ref_specs = [
        {"mask": proto_a, "color": RED},
        {"mask": proto_b, "color": BLUE},
    ]
    random.shuffle(ref_specs)

    for ref in ref_specs:
        mask = ref["mask"]
        inner_h = random.randint(max(3, mask.shape[0]), 5)
        inner_w = random.randint(max(3, mask.shape[1]), 5)
        box_h = inner_h + 2
        box_w = inner_w + 2

        placed = False
        for r0, c0 in _all_positions(h, w, box_h, box_w, margin=1):
            if not _can_place_rect(occ, r0, c0, box_h, box_w, gap=1):
                continue
            off_r = random.randint(0, inner_h - mask.shape[0])
            off_c = random.randint(0, inner_w - mask.shape[1])
            _draw_hollow_box(grid, r0, c0, box_h, box_w)
            _paste_mask(grid, mask, r0 + 1 + off_r, c0 + 1 + off_c, ref["color"])
            _mark_rect(occ, r0, c0, box_h, box_w)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place reference box")

    free_count = random.choice([4, 6])
    per_kind = free_count // 2
    kinds = ["A"] * per_kind + ["B"] * per_kind
    random.shuffle(kinds)

    free_specs: list[dict] = []
    reserve_h = max_h
    reserve_w = max_w
    for kind in kinds:
        mask = proto_a if kind == "A" else proto_b
        placed = False
        for r0, c0 in _all_positions(h, w, reserve_h, reserve_w, margin=1):
            if not _can_place_rect(occ, r0, c0, reserve_h, reserve_w, gap=1):
                continue
            _paste_mask(grid, mask, r0, c0, ORANGE)
            _mark_rect(occ, r0, c0, reserve_h, reserve_w)
            free_specs.append({"kind": kind, "r": r0, "c": c0})
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place free sprite")

    expected = _build_expected(grid, free_specs, proto_a, proto_b)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("solver mismatch on generated case")

    wrong_after_swap = _build_wrong_after_swap_original_color(grid, free_specs, proto_a, proto_b)
    wrong_no_swap = _build_wrong_no_swap(grid, free_specs, proto_a, proto_b)
    if np.array_equal(expected, wrong_after_swap):
        raise RuntimeError("degenerate case: original-shape color shortcut matches expected")
    if np.array_equal(expected, wrong_no_swap):
        raise RuntimeError("degenerate case: no-swap shortcut matches expected")

    refs = _find_reference_records(grid)
    if len(refs) != 2:
        raise RuntimeError("generated references are not uniquely recoverable")
    free_orange = _components(grid == ORANGE, connectivity=4)
    if len(free_orange) != free_count:
        raise RuntimeError("orange sprites merged or missing")
    for comp in free_orange:
        shape = _mask_from_coords(comp)
        match_count = sum(_same_mask(shape, ref["mask"]) for ref in refs)
        if match_count != 1:
            raise RuntimeError("free sprite is not uniquely classifiable")

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _generate_case()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
