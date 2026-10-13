import random
from collections import Counter

import numpy as np


BG_COLOR = 7
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def place_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        grid[int(top + rr), int(left + cc)] = int(color)


def component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def find_components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                    "color": int(color),
                }
            )
    return comps


def analyze_lockhole(mask: np.ndarray) -> dict | None:
    h, w = mask.shape
    if h < 5 or w < 5:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    candidates = []
    side_specs = {
        "top": (mask[0, 1:-1], 0),
        "bottom": (mask[-1, 1:-1], 0),
        "left": (mask[1:-1, 0], 1),
        "right": (mask[1:-1, -1], 1),
    }

    for side, (line, axis) in side_specs.items():
        missing = np.flatnonzero(~line)
        if missing.size == 0:
            continue
        if missing[-1] - missing[0] + 1 != missing.size:
            continue
        start = int(missing[0] + 1)
        length = int(missing.size)

        expected = border.copy()
        if side == "top":
            expected[0, start : start + length] = False
        elif side == "bottom":
            expected[-1, start : start + length] = False
        elif side == "left":
            expected[start : start + length, 0] = False
        else:
            expected[start : start + length, -1] = False

        if np.array_equal(mask, expected):
            candidates.append({"side": side, "start": start, "gap": length, "axis": axis})

    if len(candidates) != 1:
        return None
    return candidates[0]


def unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rot = crop_true(np.rot90(mask, k))
        key = (rot.shape, rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def orient_key(mask: np.ndarray, side: str, gap: int) -> np.ndarray:
    want_horizontal = side in ("top", "bottom")
    matches = []
    for rot in unique_rotations(mask):
        h, w = rot.shape
        if want_horizontal and w == gap and w > h:
            matches.append(rot)
        if (not want_horizontal) and h == gap and h > w:
            matches.append(rot)
    assert len(matches) == 1, "key orientation should be unambiguous"
    return matches[0]


def collect_pairs(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    pairs = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        comps = find_components_for_color(grid, color)
        frames = []
        keys = []
        for comp in comps:
            mask = component_mask(comp["coords"], comp["bbox"])
            frame = analyze_lockhole(mask)
            if frame is not None:
                frames.append((comp, frame))
            else:
                keys.append(comp)
        assert len(frames) == 1 and len(keys) == 1, "each color must give one frame and one key"
        frame_comp, frame_info = frames[0]
        key_comp = keys[0]
        pairs.append(
            {
                "color": int(color),
                "frame": frame_comp,
                "frame_info": frame_info,
                "key": key_comp,
            }
        )
    assert pairs, "expected at least one frame/key pair"
    return int(bg), pairs


def destination_for_key(frame_bbox: tuple[int, int, int, int], side: str, start: int, key_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    kh, kw = key_shape
    if side == "top":
        return int(r0 + 1), int(c0 + start)
    if side == "bottom":
        return int(r1 - kh), int(c0 + start)
    if side == "left":
        return int(r0 + start), int(c0 + 1)
    return int(r0 + start), int(c1 - kw)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, pairs = collect_pairs(g)
    out = g.copy()

    for pair in pairs:
        coords = pair["key"]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for pair in pairs:
        key_mask = component_mask(pair["key"]["coords"], pair["key"]["bbox"])
        side = pair["frame_info"]["side"]
        gap = int(pair["frame_info"]["gap"])
        start = int(pair["frame_info"]["start"])
        rotated = orient_key(key_mask, side, gap)
        dest_r, dest_c = destination_for_key(pair["frame"]["bbox"], side, start, rotated.shape)

        r0, c0, r1, c1 = pair["frame"]["bbox"]
        kh, kw = rotated.shape
        assert r0 + 1 <= dest_r <= r1 - 1
        assert c0 + 1 <= dest_c <= c1 - 1
        assert dest_r + kh - 1 <= r1 - 1
        assert dest_c + kw - 1 <= c1 - 1

        coords = np.argwhere(rotated)
        for rr, cc in coords:
            tr = int(dest_r + rr)
            tc = int(dest_c + cc)
            assert out[tr, tc] == bg
            out[tr, tc] = int(pair["color"])

    return out


def make_horizontal_key(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if height >= 3 and width >= 4:
        top = np.ones(width, dtype=bool)
        max_remove = min(3, width - 2)
        remove_count = rng.randint(0, max_remove)
        cols = list(range(1, width - 1))
        rng.shuffle(cols)
        for col in cols:
            if remove_count == 0:
                break
            if int(top.sum()) <= max(2, width // 2):
                break
            top[col] = False
            remove_count -= 1
        mask[0, :] = top
        mask[-1, :] = top[::-1]
    return mask


def build_frame_patch(height: int, width: int, color: int, bg: int, side: str, start: int, gap: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, start : start + gap] = bg
    elif side == "bottom":
        patch[-1, start : start + gap] = bg
    elif side == "left":
        patch[start : start + gap, 0] = bg
    else:
        patch[start : start + gap, -1] = bg
    return patch


def can_place_bbox(blocked: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not bool(blocked[r0:r1, c0:c1].any())


def mark_bbox(blocked: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    blocked[r0:r1, c0:c1] = True


def sample_frame_spec(color: int, rng: random.Random) -> dict:
    side = rng.choice(SIDES)
    key_depth = rng.randint(2, 4)
    gap = rng.randint(key_depth + 1, key_depth + 4)

    if side in ("top", "bottom"):
        frame_h = rng.randint(key_depth + 4, key_depth + 7)
        frame_w = rng.randint(gap + 3, gap + 6)
        start = rng.randint(1, frame_w - gap - 1)
        out_key_mask = make_horizontal_key(key_depth, gap, rng)
    else:
        frame_h = rng.randint(gap + 3, gap + 6)
        frame_w = rng.randint(key_depth + 4, key_depth + 7)
        start = rng.randint(1, frame_h - gap - 1)
        out_key_mask = np.rot90(make_horizontal_key(key_depth, gap, rng))

    frame_patch = build_frame_patch(frame_h, frame_w, color, BG_COLOR, side, start, gap)
    key_r, key_c = destination_for_key((0, 0, frame_h - 1, frame_w - 1), side, start, out_key_mask.shape)
    return {
        "color": int(color),
        "side": side,
        "gap": int(gap),
        "frame_patch": frame_patch,
        "frame_shape": (int(frame_h), int(frame_w)),
        "out_key_mask": out_key_mask,
        "inset": (int(key_r), int(key_c)),
    }


def place_frames(out: np.ndarray, specs: list[dict], rng: random.Random) -> bool:
    blocked = np.zeros(out.shape, dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["frame_shape"][0] * specs[i]["frame_shape"][1], reverse=True)

    for idx in order:
        spec = specs[idx]
        fh, fw = spec["frame_shape"]
        for _ in range(300):
            top = rng.randint(0, out.shape[0] - fh)
            left = rng.randint(0, out.shape[1] - fw)
            if not can_place_bbox(blocked, top, left, fh, fw, padding=2):
                continue
            paste_patch(out, spec["frame_patch"], top, left, BG_COLOR)
            key_top = top + spec["inset"][0]
            key_left = left + spec["inset"][1]
            place_mask(out, spec["out_key_mask"], spec["color"], key_top, key_left)
            mark_bbox(blocked, top, left, fh, fw, padding=2)
            spec["frame_pos"] = (int(top), int(left))
            spec["key_pos"] = (int(key_top), int(key_left))
            break
        else:
            return False
    return True


def scatter_keys(inp: np.ndarray, specs: list[dict], rng: random.Random) -> bool:
    blocked = np.zeros(inp.shape, dtype=bool)
    for spec in specs:
        fh, fw = spec["frame_shape"]
        top, left = spec["frame_pos"]
        mark_bbox(blocked, top, left, fh, fw, padding=1)

    order = list(range(len(specs)))
    rng.shuffle(order)
    for idx in order:
        spec = specs[idx]
        rotated = crop_true(np.rot90(spec["out_key_mask"], rng.randrange(4)))
        patch = mask_to_patch(rotated, spec["color"], BG_COLOR)
        kh, kw = patch.shape
        for _ in range(300):
            top = rng.randint(0, inp.shape[0] - kh)
            left = rng.randint(0, inp.shape[1] - kw)
            if not can_place_bbox(blocked, top, left, kh, kw, padding=1):
                continue
            paste_patch(inp, patch, top, left, BG_COLOR)
            mark_bbox(blocked, top, left, kh, kw, padding=1)
            break
        else:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_pairs = rng.randint(2, 4)
    colors = rng.sample([c for c in range(10) if c != BG_COLOR], n_pairs)
    specs = [sample_frame_spec(color, rng) for color in colors]

    base = 10 + 5 * n_pairs
    height = rng.randint(base - 2, base + 4)
    width = rng.randint(base - 2, base + 4)
    out = np.full((height, width), BG_COLOR, dtype=int)

    if not place_frames(out, specs, rng):
        return None

    inp = out.copy()
    for spec in specs:
        key_top, key_left = spec["key_pos"]
        mask = spec["out_key_mask"]
        coords = np.argwhere(mask)
        for rr, cc in coords:
            inp[int(key_top + rr), int(key_left + cc)] = BG_COLOR

    if not scatter_keys(inp, specs, rng):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
