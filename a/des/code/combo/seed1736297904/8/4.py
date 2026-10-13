from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5)) if value >= 0 else int(np.ceil(value - 0.5))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _crop_mask(mask)
    rot = np.rot90(norm, 2)
    a = ((int(norm.shape[0]), int(norm.shape[1])), norm.astype(np.uint8).tobytes())
    b = ((int(rot.shape[0]), int(rot.shape[1])), rot.astype(np.uint8).tobytes())
    return min(a, b)


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BLACK) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    keep = patch != transparent
    sub[keep] = patch[keep]
    grid[r0 : r0 + h, c0 : c0 + w] = sub


def _extract_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BLACK:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pixels = []

            while q:
                rr, cc = q.popleft()
                pixels.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.asarray(pixels, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return comps


def _aligned_top_left(anchor_bbox: tuple[int, int, int, int], move_shape: tuple[int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    anchor_center_r = (ar0 + ar1) / 2.0
    anchor_center_c = (ac0 + ac1) / 2.0
    mh, mw = move_shape
    r0 = _round_half_up(anchor_center_r - (mh - 1) / 2.0)
    c0 = _round_half_up(anchor_center_c - (mw - 1) / 2.0)
    return int(r0), int(c0)


def _select_anchor(pair: list[dict]) -> tuple[dict, dict]:
    a, b = pair
    area_a = int(a["mask"].shape[0] * a["mask"].shape[1])
    area_b = int(b["mask"].shape[0] * b["mask"].shape[1])
    if area_a < area_b:
        return b, a
    if area_b < area_a:
        return a, b

    # Under the stated shape-match rule the areas are usually equal, so row order
    # becomes the branch that actually disambiguates valid generated examples.
    if int(a["bbox"][0]) < int(b["bbox"][0]):
        return a, b
    if int(b["bbox"][0]) < int(a["bbox"][0]):
        return b, a
    if int(a["bbox"][1]) <= int(b["bbox"][1]):
        return a, b
    return b, a


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape sprites (allowing 180-degree rotation), move one sprite in each pair
    onto the other, and leave a grey shadow at the moved sprite's original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert not np.any(g == GREY), "input sprites must not use grey"

    comps = _extract_components(g)
    assert comps and len(comps) % 2 == 0, "expected an even number of foreground sprites"

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for comp in comps:
        groups.setdefault(_shape_key(comp["mask"]), []).append(comp)

    assert groups and all(len(group) == 2 for group in groups.values()), "shape matching must form unique pairs"

    out = g.copy()
    h, w = out.shape

    for pair in groups.values():
        anchor, mover = _select_anchor(pair)
        mover_patch = _paint_patch(mover["mask"], mover["color"])
        mover_coords = mover["coords"]
        out[mover_coords[:, 0], mover_coords[:, 1]] = GREY

        dest_r0, dest_c0 = _aligned_top_left(anchor["bbox"], mover["mask"].shape)
        mh, mw = mover_patch.shape
        assert 0 <= dest_r0 <= h - mh and 0 <= dest_c0 <= w - mw, "translated sprite must remain in-bounds"
        _paste(out, mover_patch, dest_r0, dest_c0, transparent=BLACK)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
PAIR_COUNT_CHOICES = [2, 3, 4]
SHAPE_BBOX_MIN = 2
SHAPE_BBOX_MAX = 5
SHAPE_CELLS_MIN = 3
SHAPE_CELLS_MAX = 8
PLACEMENT_RETRIES = 180
SHAPE_RETRIES = 300
GEN_RETRIES = 400


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _mark_mask(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask


def _shape_asymmetric(mask: np.ndarray) -> bool:
    return not np.array_equal(mask, np.rot90(mask, 2))


def _random_shape_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(SHAPE_BBOX_MIN, SHAPE_BBOX_MAX)
        w = random.randint(SHAPE_BBOX_MIN, SHAPE_BBOX_MAX)
        max_cells = min(SHAPE_CELLS_MAX, h * w - 1)
        min_cells = max(SHAPE_CELLS_MIN, (h + w) // 2)
        if min_cells > max_cells:
            continue
        target = random.randint(min_cells, max_cells)

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set()
        for dr, dc in NEIGHBORS4:
            nr, nc = seed[0] + dr, seed[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            rr, cc = cell
            mask[rr, cc] = True
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue

        mask = _crop_mask(mask)
        if int(mask.sum()) < SHAPE_CELLS_MIN:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if min(mask.shape) < 2:
            continue
        return mask

    raise RuntimeError("failed to build a valid sprite mask")


def _valid_positions(
    mask: np.ndarray,
    h: int,
    w: int,
    occupied: np.ndarray,
    *,
    min_row: int = 0,
    max_row: int | None = None,
) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    min_row = max(0, int(min_row))
    max_row = h - mh if max_row is None else min(h - mh, int(max_row))
    if min_row > max_row or mw > w:
        return []

    padded = _dilate8(occupied)
    spots = []
    for r0 in range(min_row, max_row + 1):
        for c0 in range(0, w - mw + 1):
            if not np.any(padded[r0 : r0 + mh, c0 : c0 + mw] & mask):
                spots.append((int(r0), int(c0)))
    random.shuffle(spots)
    return spots


def _build_pair_specs(pair_count: int) -> list[dict] | None:
    masks = []
    used_keys = set()
    for _ in range(pair_count):
        for _ in range(SHAPE_RETRIES):
            mask = _random_shape_mask()
            key = _shape_key(mask)
            if key in used_keys:
                continue
            used_keys.add(key)
            masks.append(mask)
            break
        else:
            return None

    colors = random.sample(PALETTE, 2 * pair_count)
    rotated_flags = [bool(random.getrandbits(1)) for _ in range(pair_count)]
    asymmetric = [i for i, mask in enumerate(masks) if _shape_asymmetric(mask)]
    if pair_count >= 2:
        if asymmetric and not any(rotated_flags[i] for i in asymmetric):
            rotated_flags[random.choice(asymmetric)] = True
        if all(rotated_flags):
            rotated_flags[random.randrange(pair_count)] = False
        if not any(rotated_flags) and asymmetric:
            rotated_flags[random.choice(asymmetric)] = True

    specs = []
    for i, mask in enumerate(masks):
        base_rot = random.randint(0, 1)
        anchor_mask = np.rot90(mask, 2 * base_rot)
        mover_mask = np.rot90(mask, 2 * (base_rot ^ int(rotated_flags[i])))
        specs.append(
            {
                "anchor_mask": anchor_mask.copy(),
                "mover_mask": mover_mask.copy(),
                "anchor_color": int(colors[2 * i]),
                "mover_color": int(colors[2 * i + 1]),
            }
        )
    return specs


def _place_pairs(pair_specs: list[dict], h: int, w: int) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placed = [dict(spec) for spec in pair_specs]
    order = sorted(
        range(len(placed)),
        key=lambda i: max(
            placed[i]["anchor_mask"].shape[0] * placed[i]["anchor_mask"].shape[1],
            placed[i]["mover_mask"].shape[0] * placed[i]["mover_mask"].shape[1],
        ),
        reverse=True,
    )

    for idx in order:
        pair = placed[idx]
        anchor_mask = pair["anchor_mask"]
        mover_mask = pair["mover_mask"]
        anchor_spots = _valid_positions(anchor_mask, h, w, occupied, max_row=h - anchor_mask.shape[0] - 2)
        if not anchor_spots:
            return None

        success = False
        for ar0, ac0 in anchor_spots:
            occupied_with_anchor = occupied.copy()
            _mark_mask(occupied_with_anchor, anchor_mask, ar0, ac0)
            mover_spots = _valid_positions(mover_mask, h, w, occupied_with_anchor, min_row=ar0 + 1)
            if not mover_spots:
                continue
            mr0, mc0 = random.choice(mover_spots)
            pair["anchor_pos"] = (int(ar0), int(ac0))
            pair["mover_pos"] = (int(mr0), int(mc0))
            occupied = occupied_with_anchor
            _mark_mask(occupied, mover_mask, mr0, mc0)
            success = True
            break

        if not success:
            return None

    return placed


def _render_input(pair_specs: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for pair in pair_specs:
        _paste(grid, _paint_patch(pair["anchor_mask"], pair["anchor_color"]), *pair["anchor_pos"])
        _paste(grid, _paint_patch(pair["mover_mask"], pair["mover_color"]), *pair["mover_pos"])
    return grid


def _expected_output(inp: np.ndarray, pair_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for pair in pair_specs:
        mr0, mc0 = pair["mover_pos"]
        mover_mask = pair["mover_mask"]
        mover_coords = np.argwhere(mover_mask) + np.array([mr0, mc0], dtype=int)
        out[mover_coords[:, 0], mover_coords[:, 1]] = GREY

        ar0, ac0 = pair["anchor_pos"]
        ah, aw = pair["anchor_mask"].shape
        anchor_bbox = (ar0, ac0, ar0 + ah - 1, ac0 + aw - 1)
        dest_r0, dest_c0 = _aligned_top_left(anchor_bbox, mover_mask.shape)
        _paste(out, _paint_patch(mover_mask, pair["mover_color"]), dest_r0, dest_c0)
    return out


def _input_is_valid(inp: np.ndarray, pair_count: int) -> bool:
    if inp.ndim != 2 or inp.size == 0:
        return False
    if np.any(inp == GREY):
        return False
    if int(np.sum(inp == BLACK)) <= int(inp.size // 2):
        return False

    comps = _extract_components(inp)
    if len(comps) != 2 * pair_count:
        return False
    counts = Counter(_shape_key(comp["mask"]) for comp in comps)
    return all(count == 2 for count in counts.values())


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    pair_specs = _build_pair_specs(pair_count)
    if pair_specs is None:
        return None

    placed = _place_pairs(pair_specs, h, w)
    if placed is None:
        return None

    inp = _render_input(placed, h, w)
    if not _input_is_valid(inp, pair_count):
        return None

    expected = _expected_output(inp, placed)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
