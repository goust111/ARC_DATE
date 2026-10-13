import math
import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != 2 * (h + w) - 4:
        return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    return r0, c0, r1, c1


def _find_frame_color(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = -1
    best_frames: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        frames: list[tuple[int, int, int, int]] = []
        for comp in _components4(grid == color):
            box = _frame_bbox(comp)
            if box is not None:
                frames.append(box)
        if len(frames) > len(best_frames):
            best_color = color
            best_frames = frames
    if not best_frames:
        raise AssertionError("expected rectangular frame components")
    return int(best_color), best_frames


def _ratio_key(h: int, w: int) -> tuple[int, int]:
    g = math.gcd(int(h), int(w))
    return int(w // g), int(h // g)


def _resize_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    src_h, src_w = mask.shape
    if (src_h, src_w) == (out_h, out_w):
        return mask.copy()
    row_idx = np.floor((np.arange(out_h) + 0.5) * src_h / out_h).astype(int)
    col_idx = np.floor((np.arange(out_w) + 0.5) * src_w / out_w).astype(int)
    row_idx = np.clip(row_idx, 0, src_h - 1)
    col_idx = np.clip(col_idx, 0, src_w - 1)
    return mask[row_idx[:, None], col_idx[None, :]]


def _rotation_from_corner(r: int, c: int, h: int, w: int) -> int:
    exact = {
        (0, 0): 0,
        (0, w - 1): 3,
        (h - 1, w - 1): 2,
        (h - 1, 0): 1,
    }
    if (r, c) in exact:
        return exact[(r, c)]
    corners = [((0, 0), 0), ((0, w - 1), 3), ((h - 1, w - 1), 2), ((h - 1, 0), 1)]
    return min(corners, key=lambda item: abs(r - item[0][0]) + abs(c - item[0][1]))[1]


def _analyze_frames(
    grid: np.ndarray, bg: int, frame_color: int, frames: list[tuple[int, int, int, int]]
) -> list[dict]:
    items: list[dict] = []
    for bbox in frames:
        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        mask = (inner != bg) & (inner != frame_color)
        coords = np.argwhere(mask)
        items.append(
            {
                "bbox": bbox,
                "inner": inner.copy(),
                "mask": mask,
                "coords": coords,
                "count": int(coords.shape[0]),
                "key": _ratio_key(inner.shape[0], inner.shape[1]),
            }
        )
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marked rectangle to the dictionary entry with the same reduced
    width:height ratio, then orient the entry's symbol by the marker corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frames = _find_frame_color(g, bg)
    infos = _analyze_frames(g, bg, frame_color, frames)

    dict_infos = [info for info in infos if info["count"] > 1]
    target_infos = [info for info in infos if info["count"] == 1]
    if not dict_infos:
        bottom = max(info["bbox"][2] for info in infos)
        dict_infos = [info for info in infos if info["bbox"][2] >= bottom - 1]
        target_infos = [info for info in infos if info not in dict_infos]

    if not dict_infos or not target_infos:
        raise AssertionError("expected both dictionary frames and target frames")

    dictionary: dict[tuple[int, int], np.ndarray] = {}
    for info in dict_infos:
        key = info["key"]
        if key not in dictionary:
            dictionary[key] = np.asarray(info["mask"], dtype=bool)

    out = g.copy()
    for info in dict_infos:
        r0, c0, r1, c1 = info["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for info in target_infos:
        key = info["key"]
        symbol = dictionary.get(key)
        if symbol is None:
            symbol = dictionary.get((key[1], key[0]))
        if symbol is None:
            raise AssertionError(f"missing dictionary entry for ratio {key}")

        inner = info["inner"]
        marker_r, marker_c = map(int, info["coords"][0])
        marker_color = int(inner[marker_r, marker_c])
        rot_k = _rotation_from_corner(marker_r, marker_c, inner.shape[0], inner.shape[1])

        # For non-square targets, 90-degree orientation swaps the source aspect
        # ratio, so rotate first and then resample back into the target interior.
        oriented = np.rot90(symbol, rot_k)
        scaled = _resize_mask(oriented, inner.shape[0], inner.shape[1])

        r0, c0, r1, c1 = info["bbox"]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[:] = bg
        patch[scaled] = marker_color

    return out


_PROTO_STRINGS = [
    ["1100", "1000", "1010", "0011"],
    ["1110", "0010", "0110", "0100"],
    ["1000", "1110", "0010", "0011"],
    ["1110", "1000", "1110", "0011"],
    ["1100", "0110", "0010", "1011"],
]

_PROTOTYPES = [np.array([[ch == "1" for ch in row] for row in rows], dtype=bool) for rows in _PROTO_STRINGS]


def _random_symbol_mask(h: int, w: int) -> np.ndarray:
    for _ in range(40):
        proto = random.choice(_PROTOTYPES)
        variant = proto.copy()
        if random.random() < 0.5:
            variant = np.fliplr(variant)
        if random.random() < 0.5:
            variant = np.flipud(variant)
        variant = np.rot90(variant, random.randrange(4))
        mask = _resize_mask(variant, h, w)
        if mask.sum() < 2 or mask.sum() >= h * w:
            continue
        variants = {_resize_mask(np.rot90(mask, k), h, w).tobytes() for k in range(4)}
        if len(variants) >= 4:
            return mask

    fallback = np.zeros((h, w), dtype=bool)
    fallback[0, : max(2, w - 1)] = True
    fallback[:, 0] = True
    fallback[h - 1, max(0, w // 2 - 1) :] = True
    if h > 2:
        fallback[h // 2 :, min(w - 1, w // 2)] = True
    return fallback


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, frame_color: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r0 + h - 1 + pad)
    cc1 = min(occupied.shape[1] - 1, c0 + w - 1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r0 + h - 1 + pad)
    cc1 = min(occupied.shape[1] - 1, c0 + w - 1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _pick_ratio_family() -> tuple[list[tuple[int, int]], dict]:
    if random.random() < 0.55:
        keys = [(1, 1), (2, 1), (1, 2)]
        cfg = {"dict_scales": [3, 4], "target_scale_range": {1: (3, 6), 2: (3, 5)}}
        return keys, cfg
    keys = [(1, 1), (3, 2), (2, 3)]
    cfg = {"dict_scales": [3], "target_scale_range": {1: (3, 5), 2: (3, 4), 3: (3, 4)}}
    return keys, cfg


def _sample_target_scale(key: tuple[int, int], cfg: dict) -> int:
    wr, hr = key
    bound = cfg["target_scale_range"][max(wr, hr)]
    return random.randint(bound[0], bound[1])


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    frame_color = 5
    palette = [1, 2, 3, 4, 6, 7, 8, 9]

    ratio_keys, cfg = _pick_ratio_family()
    dict_scale = random.choice(cfg["dict_scales"])

    dict_entries = []
    for key in ratio_keys:
        wr, hr = key
        inner_h = hr * dict_scale
        inner_w = wr * dict_scale
        mask = _random_symbol_mask(inner_h, inner_w)
        dict_entries.append({"key": key, "inner_h": inner_h, "inner_w": inner_w, "mask": mask})

    random.shuffle(dict_entries)
    dict_colors = random.sample(palette, len(dict_entries))
    gap = random.randint(1, 2)
    dict_heights = [entry["inner_h"] + 2 for entry in dict_entries]
    dict_widths = [entry["inner_w"] + 2 for entry in dict_entries]
    dict_band_h = max(dict_heights)
    dict_total_w = sum(dict_widths) + gap * (len(dict_widths) - 1)

    target_count = random.randint(4, 5)
    target_keys = [(1, 1), ratio_keys[1], ratio_keys[2], random.choice(ratio_keys)]
    if target_count == 5:
        target_keys.append(random.choice(ratio_keys))
    random.shuffle(target_keys)

    corner_cycle = [(0, 0), (0, 1), (1, 1), (1, 0)]
    corners = corner_cycle.copy()
    if target_count == 5:
        corners.append(random.choice(corner_cycle))
    random.shuffle(corners)

    target_specs = []
    for key, corner in zip(target_keys, corners):
        wr, hr = key
        scale = _sample_target_scale(key, cfg)
        inner_h = hr * scale
        inner_w = wr * scale
        target_specs.append({"key": key, "inner_h": inner_h, "inner_w": inner_w, "corner": corner})

    max_outer_w = max(spec["inner_w"] + 2 for spec in target_specs)
    max_outer_h = max(spec["inner_h"] + 2 for spec in target_specs)
    target_area = sum((spec["inner_h"] + 3) * (spec["inner_w"] + 3) for spec in target_specs)

    min_w = max(20, dict_total_w + 2, max_outer_w + 2)
    upper_h = max(max_outer_h + 2, math.ceil(target_area / max(12, min_w - 2)) + 3)
    min_h = max(20, upper_h + dict_band_h + 2)
    if min_w > 30 or min_h > 30:
        return None

    width = random.randint(min_w, 30)
    height = random.randint(min_h, 30)
    grid = np.full((height, width), bg, dtype=int)

    dict_r0 = height - dict_band_h
    start_c = random.randint(0, width - dict_total_w)
    c = start_c
    for entry, color in zip(dict_entries, dict_colors):
        outer_h = entry["inner_h"] + 2
        outer_w = entry["inner_w"] + 2
        r0 = dict_r0 + (dict_band_h - outer_h)
        _draw_frame(grid, r0, c, entry["inner_h"], entry["inner_w"], frame_color)
        inner = grid[r0 + 1 : r0 + 1 + entry["inner_h"], c + 1 : c + 1 + entry["inner_w"]]
        inner[entry["mask"]] = color
        c += outer_w + gap

    occupied = np.zeros((height, width), dtype=bool)
    _reserve(occupied, dict_r0, 0, dict_band_h, width, pad=1)

    marker_colors = random.sample(palette, len(target_specs))
    upper_limit = dict_r0 - 1
    if upper_limit <= 0:
        return None

    for spec, color in zip(target_specs, marker_colors):
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        max_r0 = upper_limit - outer_h
        if max_r0 < 0 or outer_w > width:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, width - outer_w)
            if not _fits(occupied, r0, c0, outer_h, outer_w, pad=1):
                continue
            _reserve(occupied, r0, c0, outer_h, outer_w, pad=1)
            _draw_frame(grid, r0, c0, spec["inner_h"], spec["inner_w"], frame_color)
            mr = r0 + 1 + (spec["inner_h"] - 1 if spec["corner"][0] else 0)
            mc = c0 + 1 + (spec["inner_w"] - 1 if spec["corner"][1] else 0)
            grid[mr, mc] = color
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
