from __future__ import annotations

import random
from collections import deque

import numpy as np

BG = 0
YELLOW = 4
FRAME = 9
BODY_COLORS = (1, 2, 3, 5, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_SLOT_SPAN = 3
MAX_GENERATION_ATTEMPTS = 300


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
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_to_coords(arr: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = arr[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in coords:
        mask[int(rr - r0), int(cc - c0)] = True
    return np.where(mask, sub, BG), mask


def _trim_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    best = None
    for k in range(4):
        rot = np.rot90(mask, k)
        rot = _trim_to_mask(rot.astype(int), rot)[1]
        sig = (tuple(map(int, rot.shape)), tuple(int(v) for v in rot.reshape(-1)))
        if best is None or sig < best:
            best = sig
    assert best is not None
    return best


def _marker_side(arr: np.ndarray, mask: np.ndarray) -> str:
    pts = np.argwhere((arr == YELLOW) & mask)
    assert len(pts) == 1
    r, c = map(int, pts[0])
    h, w = mask.shape
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1
    return sides[0]


def _touch_sides(bbox: tuple[int, int, int, int], frame_box: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_box
    sides = []
    if r0 == fr0 + 1:
        sides.append("top")
    if r1 == fr1 - 1:
        sides.append("bottom")
    if c0 == fc0 + 1:
        sides.append("left")
    if c1 == fc1 - 1:
        sides.append("right")
    return sides


def _assign_slot_marker(
    marker: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    slots: list[dict],
) -> tuple[int, str]:
    r, c = marker
    fr0, fc0, fr1, fc1 = frame_box
    side = None
    candidates: list[int] = []
    if r == fr0 - 1 and fc0 < c < fc1:
        side = "top"
        candidates = [
            idx
            for idx, slot in enumerate(slots)
            if "top" in slot["touches"] and slot["bbox"][1] <= c <= slot["bbox"][3]
        ]
    elif r == fr1 + 1 and fc0 < c < fc1:
        side = "bottom"
        candidates = [
            idx
            for idx, slot in enumerate(slots)
            if "bottom" in slot["touches"] and slot["bbox"][1] <= c <= slot["bbox"][3]
        ]
    elif c == fc0 - 1 and fr0 < r < fr1:
        side = "left"
        candidates = [
            idx
            for idx, slot in enumerate(slots)
            if "left" in slot["touches"] and slot["bbox"][0] <= r <= slot["bbox"][2]
        ]
    elif c == fc1 + 1 and fr0 < r < fr1:
        side = "right"
        candidates = [
            idx
            for idx, slot in enumerate(slots)
            if "right" in slot["touches"] and slot["bbox"][0] <= r <= slot["bbox"][2]
        ]
    assert side is not None and len(candidates) == 1
    return candidates[0], side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed slot with its matching outside fragment after rotating the fragment to face the slot marker.

    Input structure:
    - Background is black.
    - A large maroon rectangular frame contains several rectangular black slots separated by maroon inner walls.
    - Outside the frame are scattered monochrome fragments, each with exactly one yellow marker pixel on one edge.
    - Just outside the frame, one yellow singleton marks one outer-facing side of each slot.

    Rule:
    - Detect the maroon frame and extract every black slot inside it.
    - Match each outside fragment to the unique slot with the same shape up to 90-degree rotation.
    - Rotate the fragment so its yellow edge marker lies on the same side as that slot's yellow singleton marker, then place it into the slot.
    - Replace the fragment's yellow marker with the fragment body color, keep the maroon frame/walls, and clear everything outside the frame back to black.

    Key insight:
    - Shape determines which fragment belongs to which slot, while the yellow marker direction determines the unique rotation needed before insertion.
    """
    g = np.asarray(grid, dtype=int)
    frame_coords = np.argwhere(g == FRAME)
    assert len(frame_coords) > 0
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)
    frame_box = (fr0, fc0, fr1, fc1)

    out = np.full_like(g, BG)
    out[g == FRAME] = FRAME

    cavity = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    slot_components = _components(cavity == BG)
    slots: list[dict] = []
    for comp in slot_components:
        comp_abs = comp + np.array([fr0 + 1, fc0 + 1], dtype=int)
        bbox = _bbox(comp_abs)
        arr, mask = _crop_to_coords(np.ones_like(g, dtype=int), comp_abs)
        mask = mask.astype(bool)
        slots.append(
            {
                "bbox": bbox,
                "mask": mask,
                "touches": _touch_sides(bbox, frame_box),
                "shape": _shape_signature(mask),
            }
        )

    frame_box_mask = np.zeros_like(g, dtype=bool)
    frame_box_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    outside_components = _components((g != BG) & (~frame_box_mask))

    slot_markers: list[tuple[int, int]] = []
    fragments: list[dict] = []
    for comp in outside_components:
        vals = [int(g[int(r), int(c)]) for r, c in comp]
        if len(comp) == 1 and vals[0] == YELLOW:
            slot_markers.append((int(comp[0, 0]), int(comp[0, 1])))
            continue
        arr, mask = _crop_to_coords(g, comp)
        body_colors = {int(v) for v in arr[mask] if int(v) != YELLOW}
        assert len(body_colors) == 1
        body = int(next(iter(body_colors)))
        fragments.append(
            {
                "arr": arr,
                "mask": mask,
                "body": body,
                "shape": _shape_signature(mask),
            }
        )

    assert len(slot_markers) == len(slots) == len(fragments)
    for slot in slots:
        slot["target_side"] = None
    for marker in slot_markers:
        idx, side = _assign_slot_marker(marker, frame_box, slots)
        assert slots[idx]["target_side"] is None
        slots[idx]["target_side"] = side
    assert all(slot["target_side"] is not None for slot in slots)

    slot_by_shape: dict[tuple[tuple[int, int], tuple[int, ...]], dict] = {}
    for slot in slots:
        assert slot["shape"] not in slot_by_shape
        slot_by_shape[slot["shape"]] = slot

    for frag in fragments:
        assert frag["shape"] in slot_by_shape
        slot = slot_by_shape[frag["shape"]]
        target_side = slot["target_side"]
        sr0, sc0, sr1, sc1 = slot["bbox"]
        slot_mask = slot["mask"]
        placed = False
        for k in range(4):
            rot_arr = np.rot90(frag["arr"], k)
            rot_mask = np.rot90(frag["mask"], k)
            rot_arr, rot_mask = _trim_to_mask(rot_arr, rot_mask)
            if rot_mask.shape != slot_mask.shape or not np.array_equal(rot_mask, slot_mask):
                continue
            if _marker_side(rot_arr, rot_mask) != target_side:
                continue
            fill = rot_arr.copy()
            fill[fill == YELLOW] = frag["body"]
            out_sub = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
            out_sub[slot_mask] = fill[rot_mask]
            placed = True
            break
        assert placed

    return out


def _partition_region(
    r0: int,
    c0: int,
    h: int,
    w: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    if count == 1:
        if h < MIN_SLOT_SPAN or w < MIN_SLOT_SPAN:
            return None
        return [(r0, c0, h, w)]

    options = []
    if h >= 2 * MIN_SLOT_SPAN + 1:
        options.append("h")
    if w >= 2 * MIN_SLOT_SPAN + 1:
        options.append("v")
    if not options:
        return None

    for _ in range(40):
        orient = rng.choice(options)
        left_count = rng.randint(1, count - 1)
        right_count = count - left_count
        if orient == "h":
            cut = rng.randint(MIN_SLOT_SPAN, h - MIN_SLOT_SPAN - 1)
            top = _partition_region(r0, c0, cut, w, left_count, rng)
            bot = _partition_region(r0 + cut + 1, c0, h - cut - 1, w, right_count, rng)
            if top is not None and bot is not None:
                return top + bot
        else:
            cut = rng.randint(MIN_SLOT_SPAN, w - MIN_SLOT_SPAN - 1)
            left = _partition_region(r0, c0, h, cut, left_count, rng)
            right = _partition_region(r0, c0 + cut + 1, h, w - cut - 1, right_count, rng)
            if left is not None and right is not None:
                return left + right
    return None


def _choose_unique_partition(
    inner_h: int,
    inner_w: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(80):
        leaves = _partition_region(0, 0, inner_h, inner_w, count, rng)
        if leaves is None:
            continue
        dims = [tuple(sorted((h, w))) for _, _, h, w in leaves]
        if len(set(dims)) != len(dims):
            continue
        return leaves
    return None


def _valid_target_sides(
    leaf: tuple[int, int, int, int],
    inner_h: int,
    inner_w: int,
) -> list[str]:
    r0, c0, h, w = leaf
    sides = []
    if r0 == 0 and w >= 3:
        sides.append("top")
    if r0 + h == inner_h and w >= 3:
        sides.append("bottom")
    if c0 == 0 and h >= 3:
        sides.append("left")
    if c0 + w == inner_w and h >= 3:
        sides.append("right")
    return sides


def _slot_marker_cell(
    slot_bbox: tuple[int, int, int, int],
    side: str,
    frame_origin: tuple[int, int],
    outer_size: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = slot_bbox
    fr0, fc0 = frame_origin
    outer_h, outer_w = outer_size
    if side == "top":
        return fr0 - 1, rng.randint(sc0 + 1, sc1 - 1)
    if side == "bottom":
        return fr0 + outer_h, rng.randint(sc0 + 1, sc1 - 1)
    if side == "left":
        return rng.randint(sr0 + 1, sr1 - 1), fc0 - 1
    return rng.randint(sr0 + 1, sr1 - 1), fc0 + outer_w


def _piece_marker_index(h: int, w: int, side: str, rng: random.Random) -> tuple[int, int]:
    if side == "top":
        return 0, rng.randint(1, w - 2)
    if side == "bottom":
        return h - 1, rng.randint(1, w - 2)
    if side == "left":
        return rng.randint(1, h - 2), 0
    return rng.randint(1, h - 2), w - 1


def _make_fragment(
    h: int,
    w: int,
    target_side: str,
    body: int,
    rng: random.Random,
) -> tuple[np.ndarray, int]:
    piece = np.full((h, w), body, dtype=int)
    mr, mc = _piece_marker_index(h, w, target_side, rng)
    piece[mr, mc] = YELLOW
    rot = rng.randrange(4)
    return np.rot90(piece, rot), rot


def _can_place_rect(
    occupied: np.ndarray,
    forbidden: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    if forbidden[top : top + h, left : left + w].any():
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _build_expected(
    shape: tuple[int, int],
    frame_rect: tuple[int, int, int, int],
    slots: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_rect
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME
    for slot in slots:
        sr0, sc0, sr1, sc1 = slot["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = slot["body"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    slot_count = rng.randint(2, 4)

    leaves = None
    inner_h = inner_w = 0
    for _ in range(80):
        inner_h = rng.randint(8, 16)
        inner_w = rng.randint(8, 16)
        leaves = _choose_unique_partition(inner_h, inner_w, slot_count, rng)
        if leaves is not None and all(_valid_target_sides(leaf, inner_h, inner_w) for leaf in leaves):
            break
    if leaves is None:
        return None

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    min_h = max(outer_h + 6, 22)
    min_w = max(outer_w + 6, 22)
    if min_h > 30 or min_w > 30:
        return None

    H = rng.randint(min_h, 30)
    W = rng.randint(min_w, 30)
    top_choices = range(2, H - outer_h - 2)
    left_choices = range(2, W - outer_w - 2)
    if not top_choices or not left_choices:
        return None
    fr0 = rng.choice(tuple(top_choices))
    fc0 = rng.choice(tuple(left_choices))
    fr1 = fr0 + outer_h - 1
    fc1 = fc0 + outer_w - 1

    grid = np.full((H, W), BG, dtype=int)
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME

    slots: list[dict] = []
    for leaf in leaves:
        lr0, lc0, h, w = leaf
        sr0 = fr0 + 1 + lr0
        sc0 = fc0 + 1 + lc0
        sr1 = sr0 + h - 1
        sc1 = sc0 + w - 1
        grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
        slots.append(
            {
                "bbox": (sr0, sc0, sr1, sc1),
                "shape": tuple(sorted((h, w))),
                "h": h,
                "w": w,
            }
        )

    occupied = grid != BG
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    colors = list(BODY_COLORS)
    rng.shuffle(colors)
    for idx, slot in enumerate(slots):
        slot["body"] = colors[idx]

        rel_r0 = slot["bbox"][0] - (fr0 + 1)
        rel_c0 = slot["bbox"][1] - (fc0 + 1)
        valid_sides = _valid_target_sides((rel_r0, rel_c0, slot["h"], slot["w"]), inner_h, inner_w)
        side = rng.choice(valid_sides)
        slot["target_side"] = side
        mr, mc = _slot_marker_cell(slot["bbox"], side, (fr0, fc0), (outer_h, outer_w), rng)
        if grid[mr, mc] != BG:
            return None
        grid[mr, mc] = YELLOW
        occupied[mr, mc] = True

    pieces: list[dict] = []
    for slot in slots:
        arr, rot = _make_fragment(slot["h"], slot["w"], slot["target_side"], slot["body"], rng)
        pieces.append(
            {
                "slot": slot,
                "arr": arr,
                "area": int(arr.shape[0] * arr.shape[1]),
                "rotation": rot,
            }
        )

    pieces.sort(key=lambda piece: (-piece["area"], piece["slot"]["body"]))
    for piece in pieces:
        arr = piece["arr"]
        h, w = arr.shape
        candidates: list[tuple[int, int]] = []
        for top in range(H - h + 1):
            for left in range(W - w + 1):
                if _can_place_rect(occupied, forbidden, top, left, h, w):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        grid[top : top + h, left : left + w] = arr
        occupied[top : top + h, left : left + w] = True

    expected = _build_expected((H, W), (fr0, fc0, fr1, fc1), slots)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-slot puzzle")
