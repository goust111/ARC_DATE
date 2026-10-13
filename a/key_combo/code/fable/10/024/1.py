import random
from typing import Optional

import numpy as np


BG = 0
BLUE = 1
RED = 2
TEAL = 8
FRAME_COLORS = (3, 4, 5, 6, 7, 9)
SIDES = ("top", "right", "bottom", "left")
ROT_TO_TOP = {"top": 0, "right": 1, "bottom": 2, "left": 3}


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _guide_mask(size: int, side: str) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "right":
        mask[:, -1] = True
    elif side == "bottom":
        mask[-1, :] = True
    elif side == "left":
        mask[:, 0] = True
    else:
        raise ValueError(f"unknown side: {side}")
    return mask


def _rotate_to_top(device: np.ndarray, guide_side: str) -> np.ndarray:
    return np.rot90(device, ROT_TO_TOP[guide_side])


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _parse_device(sub: np.ndarray) -> Optional[dict]:
    if sub.ndim != 2 or sub.shape[0] != sub.shape[1]:
        return None
    size = int(sub.shape[0])
    if size < 5:
        return None

    guide_sides = [side for side in SIDES if np.all(sub[_guide_mask(size, side)] == TEAL)]
    if len(guide_sides) != 1:
        return None
    guide_side = guide_sides[0]

    border = _border_mask(size)
    frame_cells = border & (~_guide_mask(size, guide_side))
    frame_vals = np.unique(sub[frame_cells])
    if len(frame_vals) != 1:
        return None
    frame_color = int(frame_vals[0])
    if frame_color in (BG, BLUE, TEAL):
        return None

    interior = sub[1:-1, 1:-1]
    if interior.size == 0:
        return None
    if not set(map(int, np.unique(interior))).issubset({BG, BLUE}):
        return None

    if not np.all(np.isin(sub[border], [frame_color, TEAL])):
        return None

    canonical = _rotate_to_top(sub, guide_side)
    return {
        "guide_side": guide_side,
        "frame_color": frame_color,
        "raw_mask": sub[1:-1, 1:-1] == BLUE,
        "mask": canonical[1:-1, 1:-1] == BLUE,
        "size": size,
    }


def _find_devices(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    seen = set()
    for size in range(5, min(h, w) + 1):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
                if bbox in seen:
                    continue
                sub = g[r0 : r0 + size, c0 : c0 + size]
                parsed = _parse_device(sub)
                if parsed is None:
                    continue
                parsed["bbox"] = bbox
                candidates.append(parsed)
                seen.add(bbox)
    candidates.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["size"]))
    if len(candidates) != 2:
        raise ValueError(f"expected exactly 2 devices, found {len(candidates)}")
    if candidates[0]["size"] != candidates[1]["size"]:
        raise ValueError("devices must have the same outer size")
    if _boxes_overlap(candidates[0]["bbox"], candidates[1]["bbox"], padding=0):
        raise ValueError("devices must not overlap")
    if candidates[0]["frame_color"] == candidates[1]["frame_color"]:
        raise ValueError("devices must use distinct frame colors")
    return candidates


def _compare_masks(mask_a: np.ndarray, mask_b: np.ndarray, frame_a: int, frame_b: int) -> np.ndarray:
    out = np.zeros(mask_a.shape, dtype=int)
    both = mask_a & mask_b
    only_a = mask_a & (~mask_b)
    only_b = mask_b & (~mask_a)
    out[both] = RED
    out[only_a] = int(frame_a)
    out[only_b] = int(frame_b)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate two framed bitmask devices to a common guide-edge orientation, then color their cellwise overlap by source ownership.

    Input structure:
    - A black canvas contains exactly two disjoint square devices.
    - Each device has a one-cell border ring: one full border side is teal as a guide edge, and the other border cells use that device's own frame color.
    - Inside each frame is a same-sized blue/black bitmask; the two devices may appear in different rotations.

    Rule:
    - For each device, rotate the whole square by a multiple of 90 degrees so its teal guide edge becomes the top side.
    - Read the two aligned interior bitmasks cell by cell.
    - Output a k x k grid where blue/blue becomes red, black/black stays black, and a blue cell present in only one device takes that device's frame color.

    Key insight:
    - The visible canvas orientation is not the comparison frame: the teal guide edge defines each device's own "up", so you must canonicalize both devices before the logical comparison and attribution coloring.
    """
    g = np.asarray(grid, dtype=int)
    devices = _find_devices(g)
    mask_a = devices[0]["mask"]
    mask_b = devices[1]["mask"]
    return _compare_masks(mask_a, mask_b, devices[0]["frame_color"], devices[1]["frame_color"])


def _stamp_rect(mask: np.ndarray) -> None:
    k = mask.shape[0]
    hh = random.randint(1, min(3, k))
    ww = random.randint(1, min(3, k))
    r0 = random.randint(0, k - hh)
    c0 = random.randint(0, k - ww)
    mask[r0 : r0 + hh, c0 : c0 + ww] = True


def _stamp_line(mask: np.ndarray) -> None:
    k = mask.shape[0]
    length = random.randint(2, k)
    if random.random() < 0.5:
        r = random.randint(0, k - 1)
        c0 = random.randint(0, k - length)
        mask[r, c0 : c0 + length] = True
    else:
        c = random.randint(0, k - 1)
        r0 = random.randint(0, k - length)
        mask[r0 : r0 + length, c] = True


def _stamp_l(mask: np.ndarray) -> None:
    k = mask.shape[0]
    arm = random.randint(2, max(2, min(4, k)))
    r0 = random.randint(0, k - arm)
    c0 = random.randint(0, k - arm)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[r0, c0 : c0 + arm] = True
        mask[r0 : r0 + arm, c0] = True
    elif corner == "tr":
        mask[r0, c0 : c0 + arm] = True
        mask[r0 : r0 + arm, c0 + arm - 1] = True
    elif corner == "bl":
        mask[r0 + arm - 1, c0 : c0 + arm] = True
        mask[r0 : r0 + arm, c0] = True
    else:
        mask[r0 + arm - 1, c0 : c0 + arm] = True
        mask[r0 : r0 + arm, c0 + arm - 1] = True


def _stamp_diag(mask: np.ndarray) -> None:
    k = mask.shape[0]
    length = random.randint(2, max(2, min(5, k)))
    r0 = random.randint(0, k - length)
    c0 = random.randint(0, k - length)
    if random.random() < 0.5:
        for i in range(length):
            mask[r0 + i, c0 + i] = True
    else:
        for i in range(length):
            mask[r0 + i, c0 + length - 1 - i] = True


def _random_mask(k: int) -> np.ndarray:
    area = k * k
    for _ in range(200):
        mask = np.zeros((k, k), dtype=bool)
        n_shapes = random.randint(2, 4)
        for _ in range(n_shapes):
            random.choice((_stamp_rect, _stamp_line, _stamp_l, _stamp_diag))(mask)
        if random.random() < 0.4:
            dots = random.randint(1, max(1, k // 2))
            for _ in range(dots):
                mask[random.randint(0, k - 1), random.randint(0, k - 1)] = True
        blue_count = int(mask.sum())
        if 3 <= blue_count <= area - 3:
            return mask
    raise RuntimeError("failed to build a non-degenerate mask")


def _build_device(mask: np.ndarray, frame_color: int) -> np.ndarray:
    k = mask.shape[0]
    out = np.full((k + 2, k + 2), frame_color, dtype=int)
    out[0, :] = TEAL
    out[1:-1, 1:-1] = np.where(mask, BLUE, BG)
    return out


def _input_oriented_device(canonical_device: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(canonical_device, -int(turns_cw))


def _status_coverage(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int, int, int]:
    both_black = int((~mask_a & ~mask_b).sum())
    only_a = int((mask_a & ~mask_b).sum())
    only_b = int((~mask_a & mask_b).sum())
    both_blue = int((mask_a & mask_b).sum())
    return both_black, only_a, only_b, both_blue


def _place_devices(dev_a: np.ndarray, dev_b: np.ndarray) -> np.ndarray:
    ha, wa = dev_a.shape
    hb, wb = dev_b.shape
    min_h = max(ha, hb) + 4
    min_w = max(wa, wb) + 4
    for _ in range(300):
        h = random.randint(max(min_h, ha + hb + 2), 30)
        w = random.randint(max(min_w, wa + wb + 2), 30)
        grid = np.zeros((h, w), dtype=int)
        r0a = random.randint(1, h - ha - 1)
        c0a = random.randint(1, w - wa - 1)
        bbox_a = (r0a, c0a, r0a + ha - 1, c0a + wa - 1)
        for _ in range(200):
            r0b = random.randint(1, h - hb - 1)
            c0b = random.randint(1, w - wb - 1)
            bbox_b = (r0b, c0b, r0b + hb - 1, c0b + wb - 1)
            if _boxes_overlap(bbox_a, bbox_b, padding=1):
                continue
            grid[r0a : r0a + ha, c0a : c0a + wa] = np.where(dev_a != BG, dev_a, grid[r0a : r0a + ha, c0a : c0a + wa])
            grid[r0b : r0b + hb, c0b : c0b + wb] = np.where(dev_b != BG, dev_b, grid[r0b : r0b + hb, c0b : c0b + wb])
            return grid
    raise RuntimeError("failed to place devices")


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        devices = _find_devices(grid)
    except Exception:
        return False

    mask_a = devices[0]["mask"]
    mask_b = devices[1]["mask"]
    raw_a = devices[0]["raw_mask"]
    raw_b = devices[1]["raw_mask"]
    frame_a = devices[0]["frame_color"]
    frame_b = devices[1]["frame_color"]

    counts = _status_coverage(mask_a, mask_b)
    if min(counts) <= 0:
        return False

    correct = _compare_masks(mask_a, mask_b, frame_a, frame_b)
    raw_compare = _compare_masks(raw_a, raw_b, frame_a, frame_b)
    if np.array_equal(correct, raw_compare):
        return False

    colors = set(map(int, np.unique(correct)))
    if colors != {BG, RED, frame_a, frame_b}:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, correct)


def generate_input() -> np.ndarray:
    for _ in range(500):
        k = random.randint(4, 8)
        frame_a, frame_b = random.sample(FRAME_COLORS, 2)

        mask_a = _random_mask(k)
        mask_b = _random_mask(k)
        if min(_status_coverage(mask_a, mask_b)) <= 0:
            continue

        dev_a = _build_device(mask_a, frame_a)
        dev_b = _build_device(mask_b, frame_b)

        turns_a = random.randint(0, 3)
        turns_b = random.randint(0, 3)
        inp_a = _input_oriented_device(dev_a, turns_a)
        inp_b = _input_oriented_device(dev_b, turns_b)

        grid = _place_devices(inp_a, inp_b)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
