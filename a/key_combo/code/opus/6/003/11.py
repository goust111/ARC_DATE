from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME = 5

GRID_MIN = 20
GRID_MAX = 30
FRAME_SIDES = (5, 7, 9)
FRAME_COUNT_RANGE = (4, 6)
MAX_GENERATION_ATTEMPTS = 500

_BASE_PROTOTYPES = (
    np.array([[1, 0, 2], [2, 1, 0], [1, 2, 1]], dtype=int),
    np.array([[2, 1, 2], [0, 2, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 2, 0], [0, 1, 2], [2, 1, 1]], dtype=int),
    np.array([[2, 0, 1], [1, 2, 2], [0, 1, 2]], dtype=int),
)


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _rounded_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 1:-1] = True
    mask[h - 1, 1:-1] = True
    mask[1:-1, 0] = True
    mask[1:-1, w - 1] = True
    return mask


def _sample_indices(src_len: int, dst_len: int) -> list[int]:
    if dst_len == 1:
        return [0]
    if src_len == 1:
        return [0] * dst_len
    return [int(round(i * (src_len - 1) / (dst_len - 1))) for i in range(dst_len)]


def _resize_nn(arr: np.ndarray, out_h: int = 3, out_w: int = 3) -> np.ndarray:
    rows = _sample_indices(arr.shape[0], out_h)
    cols = _sample_indices(arr.shape[1], out_w)
    return arr[np.ix_(rows, cols)].copy()


def _expand_from_proto(proto: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    row_samples = _sample_indices(out_h, proto.shape[0])
    col_samples = _sample_indices(out_w, proto.shape[1])
    row_bins = [min(range(proto.shape[0]), key=lambda i: (abs(r - row_samples[i]), i)) for r in range(out_h)]
    col_bins = [min(range(proto.shape[1]), key=lambda i: (abs(c - col_samples[i]), i)) for c in range(out_w)]
    out = np.zeros((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            out[r, c] = int(proto[row_bins[r], col_bins[c]])
    return out


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != bg)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in components_mask(grid == FRAME, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5 or h > 9 or w > 9:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = _rounded_frame_mask(h, w)
        if not np.array_equal(local, expected):
            continue
        if (
            int(grid[r0, c0]) != BG
            or int(grid[r0, c1]) != BG
            or int(grid[r1, c0]) != BG
            or int(grid[r1, c1]) != BG
        ):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or np.any(inner == FRAME):
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "inner": inner.copy()})
    frames.sort(key=lambda rec: rec["bbox"])
    return frames


def _normalize_frame_content(inner: np.ndarray, bg: int) -> np.ndarray:
    crop = _crop_non_bg(inner, bg)
    return _resize_nn(crop, 3, 3)


def _stamp(proto: np.ndarray) -> np.ndarray:
    patch = np.zeros((5, 5), dtype=int)
    patch[_rounded_frame_mask(5, 5)] = FRAME
    patch[1:4, 1:4] = proto
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each rounded frame with a centered standard stamp carrying the opposite normalized prototype.

    Input structure:
    - The background is black, and 4 to 6 separated grey rounded-rectangle frames are scattered on the canvas.
    - Each frame has black corners, grey outer border pixels on the remaining perimeter, and a blue/red interior pattern inside.
    - Frame sizes vary, and each interior pattern may occupy only part of the inner area before being cropped.

    Rule:
    - Detect every grey rounded frame and take only its inner area.
    - Crop the non-background interior pattern to its own bounding box, then resize that cropped pattern to 3x3 by nearest-neighbor sampling.
    - These normalized 3x3 patterns form exactly two prototype groups; for each frame, choose the other group's prototype.
    - Output a black canvas of the same size and place a standard 5x5 rounded frame at the original frame center, filling its 3x3 interior with the opposite normalized prototype.

    Key insight:
    - The same logical motif appears at different resolutions and with extra padding inside larger frames, so cropping and normalization must happen before grouping and swapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    frames = _detect_frames(g)
    assert 4 <= len(frames) <= 6

    groups: dict[tuple[int, ...], list[int]] = {}
    norms: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        norm = _normalize_frame_content(frame["inner"], bg)
        norms.append(norm)
        key = tuple(map(int, norm.flat))
        groups.setdefault(key, []).append(idx)

    assert len(groups) == 2
    assert all(len(indices) >= 2 for indices in groups.values())

    keys = list(groups.keys())
    swap = {
        keys[0]: np.array(keys[1], dtype=int).reshape(3, 3),
        keys[1]: np.array(keys[0], dtype=int).reshape(3, 3),
    }

    out = np.full_like(g, bg)
    for frame, norm in zip(frames, norms, strict=False):
        r0, c0, r1, c1 = frame["bbox"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        top = center_r - 2
        left = center_c - 2
        paste(out, _stamp(swap[tuple(map(int, norm.flat))]), top, left)
    return out


def _random_transform(proto: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(proto, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, k=rng.randrange(4)).copy()


def _choose_prototypes(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    idxs = list(range(len(_BASE_PROTOTYPES)))
    rng.shuffle(idxs)
    a = _random_transform(_BASE_PROTOTYPES[idxs[0]], rng)
    b = _random_transform(_BASE_PROTOTYPES[idxs[1]], rng)
    if np.array_equal(a, b):
        b = _random_transform(_BASE_PROTOTYPES[idxs[2]], rng)
    assert not np.array_equal(a, b)
    return a, b


def _build_frame_patch(
    proto: np.ndarray,
    outer_h: int,
    outer_w: int,
    crop_h: int,
    crop_w: int,
    margin_top: int,
    margin_left: int,
) -> np.ndarray:
    patch = np.zeros((outer_h, outer_w), dtype=int)
    patch[_rounded_frame_mask(outer_h, outer_w)] = FRAME
    interior = np.zeros((outer_h - 2, outer_w - 2), dtype=int)
    crop = _expand_from_proto(proto, crop_h, crop_w)
    interior[margin_top : margin_top + crop_h, margin_left : margin_left + crop_w] = crop
    patch[1:-1, 1:-1] = interior
    return patch


def _place_frames(
    rng: random.Random,
    shape: tuple[int, int],
    sizes: list[tuple[int, int]],
    gap: int = 1,
) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        ph, pw = sizes[idx]
        placed = False
        for _ in range(80):
            r0 = rng.randint(0, h - ph)
            c0 = rng.randint(0, w - pw)
            rr0 = max(0, r0 - gap)
            rr1 = min(h, r0 + ph + gap)
            cc0 = max(0, c0 - gap)
            cc1 = min(w, c0 + pw + gap)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _expected_output(shape: tuple[int, int], records: list[dict], protos: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for rec in records:
        r0, c0 = rec["pos"]
        oh, ow = rec["outer"]
        center_r = r0 + oh // 2
        center_c = c0 + ow // 2
        top = center_r - 2
        left = center_c - 2
        paste(out, _stamp(protos[1 - rec["group"]]), top, left)
    return out


def _full_inner_normalized(record: dict, proto: np.ndarray) -> np.ndarray:
    inner_h = record["outer"][0] - 2
    inner_w = record["outer"][1] - 2
    full_inner = np.zeros((inner_h, inner_w), dtype=int)
    crop = _expand_from_proto(proto, record["crop"][0], record["crop"][1])
    mt, ml = record["margin"]
    full_inner[mt : mt + crop.shape[0], ml : ml + crop.shape[1]] = crop
    return _resize_nn(full_inner, 3, 3)


def _semantic_checks(grid: np.ndarray, records: list[dict], protos: tuple[np.ndarray, np.ndarray], expected: np.ndarray) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    by_group: dict[int, list[dict]] = {0: [], 1: []}
    for rec in records:
        by_group[rec["group"]].append(rec)

    outer_by_group = [{tuple(rec["outer"]) for rec in by_group[g]} for g in (0, 1)]
    if not (outer_by_group[0] & outer_by_group[1]):
        return False

    crop_varies = any(len({tuple(rec["crop"]) for rec in by_group[g]}) >= 2 for g in (0, 1))
    if not crop_varies:
        return False

    crop_needed = False
    for rec in records:
        proto = protos[rec["group"]]
        if tuple(rec["margin"]) == (0, 0) and rec["crop"] == (rec["outer"][0] - 2, rec["outer"][1] - 2):
            continue
        if not np.array_equal(_full_inner_normalized(rec, proto), proto):
            crop_needed = True
            break
    if not crop_needed:
        return False

    resized_needed = any(rec["crop"][0] > 3 or rec["crop"][1] > 3 for rec in records)
    if not resized_needed:
        return False

    solver_frames = _detect_frames(grid)
    if len(solver_frames) != len(records):
        return False

    solver_groups = {}
    for frame in solver_frames:
        key = tuple(map(int, _normalize_frame_content(frame["inner"], BG).flat))
        solver_groups[key] = solver_groups.get(key, 0) + 1
    if sorted(solver_groups.values()) != sorted(len(by_group[g]) for g in (0, 1)):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    frame_count = rng.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    count_a = rng.randint(2, frame_count - 2)
    count_b = frame_count - count_a
    labels = [0] * count_a + [1] * count_b
    rng.shuffle(labels)

    protos = _choose_prototypes(rng)
    shared_outer = (rng.choice(FRAME_SIDES), rng.choice(FRAME_SIDES))
    a_idx = labels.index(0)
    b_idx = labels.index(1)

    records: list[dict] = []
    for idx, group in enumerate(labels):
        if idx in (a_idx, b_idx):
            outer = shared_outer
        else:
            outer = (rng.choice(FRAME_SIDES), rng.choice(FRAME_SIDES))
        inner_h = outer[0] - 2
        inner_w = outer[1] - 2

        crop_h = rng.randint(3, inner_h)
        crop_w = rng.randint(3, inner_w)
        if inner_h > 3 and rng.random() < 0.75:
            crop_h = rng.randint(4, inner_h)
        if inner_w > 3 and rng.random() < 0.75:
            crop_w = rng.randint(4, inner_w)

        max_top = inner_h - crop_h
        max_left = inner_w - crop_w
        margin_top = rng.randint(0, max_top) if max_top > 0 else 0
        margin_left = rng.randint(0, max_left) if max_left > 0 else 0
        if (max_top > 0 or max_left > 0) and rng.random() < 0.75:
            if max_top > 0:
                margin_top = rng.randint(0, max_top)
            if max_left > 0:
                margin_left = rng.randint(0, max_left)

        records.append(
            {
                "group": group,
                "outer": outer,
                "crop": (crop_h, crop_w),
                "margin": (margin_top, margin_left),
            }
        )

    placements = _place_frames(rng, (grid_h, grid_w), [rec["outer"] for rec in records], gap=1)
    if placements is None or len(placements) != len(records):
        return None
    for rec, pos in zip(records, placements, strict=False):
        rec["pos"] = pos

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for rec in records:
        proto = protos[rec["group"]]
        patch = _build_frame_patch(
            proto,
            rec["outer"][0],
            rec["outer"][1],
            rec["crop"][0],
            rec["crop"][1],
            rec["margin"][0],
            rec["margin"][1],
        )
        paste(grid, patch, rec["pos"][0], rec["pos"][1])

    expected = _expected_output(grid.shape, records, protos)
    if not _semantic_checks(grid, records, protos, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rounded-frame prototype-swap puzzle")
