import random
from typing import List, Optional, Sequence, Tuple

import numpy as np


BG = 8
FRAME = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 300
SIDES = ("up", "down", "left", "right")


def _components(mask: np.ndarray, connectivity: int = 4) -> List[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1 - 1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _compress_cycle(values: Sequence[int]) -> List[int]:
    runs: List[int] = []
    for value in values:
        value = int(value)
        if not runs or runs[-1] != value:
            runs.append(value)
    if len(runs) > 1 and runs[0] == runs[-1]:
        runs.pop()
    return runs


def _extract_stamp_layers(patch: np.ndarray) -> Optional[List[np.ndarray]]:
    h, w = patch.shape
    top, left = 0, 0
    bottom, right = h - 1, w - 1
    masks: List[np.ndarray] = []

    while top <= bottom and left <= right:
        sub = patch[top : bottom + 1, left : right + 1]
        if np.all(sub == sub[0, 0]):
            mask = np.zeros((h, w), dtype=bool)
            mask[top : bottom + 1, left : right + 1] = True
            masks.append(mask)
            return masks

        border_mask = np.zeros(sub.shape, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_values = np.unique(sub[border_mask])
        if border_values.size != 1:
            return None

        mask = np.zeros((h, w), dtype=bool)
        mask[top, left : right + 1] = True
        mask[bottom, left : right + 1] = True
        mask[top : bottom + 1, left] = True
        mask[top : bottom + 1, right] = True
        masks.append(mask)

        top += 1
        left += 1
        bottom -= 1
        right -= 1

    return masks if masks else None


def _paint_layers(shape: Tuple[int, int], colors: Sequence[int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    top, left = 0, 0
    bottom, right = h - 1, w - 1

    for index, color in enumerate(colors):
        color = int(color)
        if index == len(colors) - 1:
            out[top : bottom + 1, left : right + 1] = color
        else:
            out[top, left : right + 1] = color
            out[bottom, left : right + 1] = color
            out[top : bottom + 1, left] = color
            out[top : bottom + 1, right] = color
            top += 1
            left += 1
            bottom -= 1
            right -= 1
    return out


def _find_frames(grid: np.ndarray) -> List[dict]:
    frames = []
    for coords in _components(grid == FRAME, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 5 or w < 5:
            continue

        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if not np.array_equal(sub == FRAME, expected):
            continue

        cavity = sub[2:-2, 2:-2]
        if cavity.size == 0 or not np.all(cavity == BG):
            continue

        band_path = _perimeter_coords(1, 1, h - 2, w - 2)
        band_values = [int(sub[r, c]) for r, c in band_path]
        if any(value in (BG, FRAME) for value in band_values):
            continue

        band_colors = _compress_cycle(band_values)
        if not 2 <= len(band_colors) <= 4:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "band_colors": band_colors,
                "cavity_shape": tuple(map(int, cavity.shape)),
            }
        )

    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_stamps(grid: np.ndarray) -> List[dict]:
    stamps = []
    mask = (grid != BG) & (grid != FRAME)
    for coords in _components(mask, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if coords.shape[0] != h * w:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        layers = _extract_stamp_layers(patch)
        if layers is None or not 2 <= len(layers) <= 4:
            continue

        stamps.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (h, w),
                "layers": layers,
                "layer_count": len(layers),
            }
        )

    stamps.sort(key=lambda item: item["bbox"])
    return stamps


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _select_stamp(frame: dict, stamps: List[dict], used: set) -> Optional[int]:
    candidates = [idx for idx, stamp in enumerate(stamps) if idx not in used and stamp["layer_count"] == len(frame["band_colors"])]
    if not candidates:
        return None

    exact_shape = [idx for idx in candidates if tuple(stamps[idx]["shape"]) == tuple(frame["cavity_shape"])]
    if exact_shape:
        candidates = exact_shape

    fr_center = _bbox_center(frame["bbox"])

    def score(index: int) -> Tuple[float, Tuple[int, int, int, int]]:
        sr_center = _bbox_center(stamps[index]["bbox"])
        dist = abs(fr_center[0] - sr_center[0]) + abs(fr_center[1] - sr_center[1])
        return dist, stamps[index]["bbox"]

    candidates.sort(key=score)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    stamps = _find_stamps(g)
    used: set = set()

    for frame in frames:
        stamp_index = _select_stamp(frame, stamps, used)
        if stamp_index is None:
            continue
        used.add(stamp_index)
        stamp = stamps[stamp_index]

        sr0, sc0, sr1, sc1 = stamp["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG

        recolored = np.zeros(tuple(stamp["shape"]), dtype=int)
        for mask, color in zip(stamp["layers"], frame["band_colors"]):
            recolored[mask] = int(color)

        fr0, fc0, fr1, fc1 = frame["bbox"]
        out[fr0 + 2 : fr1 - 1, fc0 + 2 : fc1 - 1] = recolored

    return out


def _random_partition(total: int, parts: int, min_part: int = 1) -> List[int]:
    slots = [min_part] * parts
    extra = total - parts * min_part
    for _ in range(extra):
        slots[random.randrange(parts)] += 1
    random.shuffle(slots)
    return slots


def _choose_colors(count: int) -> Tuple[List[int], List[int]]:
    chosen = random.sample(COLOR_POOL, 2 * count)
    band = chosen[:count]
    stamp = chosen[count:]
    return band, stamp


def _build_frame(cavity_h: int, cavity_w: int, band_colors: Sequence[int]) -> np.ndarray:
    h = cavity_h + 4
    w = cavity_w + 4
    patch = np.full((h, w), BG, dtype=int)

    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME

    path = _perimeter_coords(1, 1, h - 2, w - 2)
    lengths = _random_partition(len(path), len(band_colors), min_part=2)
    index = 0
    for color, run_length in zip(band_colors, lengths):
        for _ in range(run_length):
            r, c = path[index]
            patch[r, c] = int(color)
            index += 1
    return patch


def _build_stamp(shape: Tuple[int, int], colors: Sequence[int]) -> np.ndarray:
    return _paint_layers(shape, colors)


def _crop_non_bg(patch: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    coords = np.argwhere(patch != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = patch[r0 : r1 + 1, c0 : c1 + 1]
    return cropped, (int(r0), int(c0))


def _compose_cluster(frame_patch: np.ndarray, stamp_patch: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    fh, fw = frame_patch.shape
    sh, sw = stamp_patch.shape
    side = random.choice(SIDES)
    gap = random.randint(1, 3)
    slack = random.randint(0, 2)

    if side in ("up", "down"):
        total_h = fh + sh + gap
        total_w = max(fw, sw) + slack
        fc = random.randint(0, total_w - fw)
        sc = random.randint(0, total_w - sw)
        if side == "up":
            sr, fr = 0, sh + gap
        else:
            fr, sr = 0, fh + gap
    else:
        total_w = fw + sw + gap
        total_h = max(fh, sh) + slack
        fr = random.randint(0, total_h - fh)
        sr = random.randint(0, total_h - sh)
        if side == "left":
            sc, fc = 0, sw + gap
        else:
            fc, sc = 0, fw + gap

    patch = np.full((total_h, total_w), BG, dtype=int)
    patch[fr : fr + fh, fc : fc + fw] = frame_patch
    patch[sr : sr + sh, sc : sc + sw] = stamp_patch

    cropped, (off_r, off_c) = _crop_non_bg(patch)
    fr -= off_r
    fc -= off_c
    sr -= off_r
    sc -= off_c

    frame_bbox = (fr, fc, fr + fh - 1, fc + fw - 1)
    stamp_bbox = (sr, sc, sr + sh - 1, sc + sw - 1)
    return cropped, frame_bbox, stamp_bbox


def _translate_bbox(bbox: Tuple[int, int, int, int], dr: int, dc: int) -> Tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != BG, patch, grid[r0 : r0 + h, c0 : c0 + w])


def _place_clusters(clusters: List[dict]) -> Optional[Tuple[np.ndarray, List[dict]]]:
    padded_area = sum((cluster["patch"].shape[0] + 2) * (cluster["patch"].shape[1] + 2) for cluster in clusters)
    max_h = max(cluster["patch"].shape[0] for cluster in clusters)
    max_w = max(cluster["patch"].shape[1] for cluster in clusters)
    side_est = int(np.ceil(np.sqrt(padded_area * 1.6)))

    min_h = max(22, max_h + 2, side_est)
    min_w = max(22, max_w + 2, side_est)
    if min_h > 42 or min_w > 42:
        return None

    grid_h = random.randint(min_h, min(42, min_h + 6))
    grid_w = random.randint(min_w, min(42, min_w + 6))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    placed = []
    order = sorted(clusters, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)
    for cluster in order:
        patch = cluster["patch"]
        ch, cw = patch.shape
        if ch + 2 > grid_h or cw + 2 > grid_w:
            return None

        found = False
        for _ in range(250):
            r0 = random.randint(1, grid_h - ch - 1)
            c0 = random.randint(1, grid_w - cw - 1)
            r_start = max(0, r0 - 1)
            c_start = max(0, c0 - 1)
            r_end = min(grid_h, r0 + ch + 1)
            c_end = min(grid_w, c0 + cw + 1)
            if occupied[r_start:r_end, c_start:c_end].any():
                continue

            _paste(grid, patch, r0, c0)
            occupied[r0 : r0 + ch, c0 : c0 + cw] = True
            placed.append(
                {
                    "frame_bbox": _translate_bbox(cluster["frame_bbox"], r0, c0),
                    "stamp_bbox": _translate_bbox(cluster["stamp_bbox"], r0, c0),
                    "band_colors": list(cluster["band_colors"]),
                    "cavity_shape": tuple(cluster["cavity_shape"]),
                }
            )
            found = True
            break
        if not found:
            return None

    return grid, placed


def _expected_output(grid: np.ndarray, placements: List[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        sr0, sc0, sr1, sc1 = item["stamp_bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG

        fr0, fc0, fr1, fc1 = item["frame_bbox"]
        recolored = _build_stamp(tuple(item["cavity_shape"]), item["band_colors"])
        out[fr0 + 2 : fr1 - 1, fc0 + 2 : fc1 - 1] = recolored
    return out


def _gen_attempt() -> Optional[np.ndarray]:
    object_count = random.randint(1, 3)
    layer_counts = random.sample([2, 3, 4], object_count)
    clusters = []

    for layer_count in layer_counts:
        min_side = 2 * layer_count - 1
        cavity_h = random.randint(min_side, min_side + 3)
        cavity_w = random.randint(min_side, min_side + 3)
        band_colors, stamp_colors = _choose_colors(layer_count)

        frame_patch = _build_frame(cavity_h, cavity_w, band_colors)
        stamp_patch = _build_stamp((cavity_h, cavity_w), stamp_colors)
        cluster_patch, frame_bbox, stamp_bbox = _compose_cluster(frame_patch, stamp_patch)
        clusters.append(
            {
                "patch": cluster_patch,
                "frame_bbox": frame_bbox,
                "stamp_bbox": stamp_bbox,
                "band_colors": list(band_colors),
                "cavity_shape": (cavity_h, cavity_w),
            }
        )

    placed = _place_clusters(clusters)
    if placed is None:
        return None

    grid, placement_info = placed
    expected = _expected_output(grid, placement_info)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
