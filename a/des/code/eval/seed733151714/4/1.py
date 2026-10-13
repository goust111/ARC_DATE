import random
from typing import Dict, List, Optional, Tuple

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((0, 0), dtype=bool)
    return mask[rows][:, cols]


def _bbox_from_coords(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    if bbox is None:
        bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _monochrome_components(grid: np.ndarray, bg: int) -> List[Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[Dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": _coords_to_mask(coords, bbox),
                }
            )
    return comps


def _rotation_variants(mask: np.ndarray) -> List[np.ndarray]:
    uniq: List[np.ndarray] = []
    seen = set()
    for k in range(4):
        rotated = _trim_mask(np.rot90(mask, k))
        key = (rotated.shape, rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(rotated)
    return uniq


def _shape_key(mask: np.ndarray) -> Tuple[int, int, bytes]:
    best: Optional[Tuple[int, int, bytes]] = None
    for rotated in _rotation_variants(mask):
        key = (int(rotated.shape[0]), int(rotated.shape[1]), rotated.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _find_frames(grid: np.ndarray, bg: int) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for comp in _monochrome_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue
        mask = comp["mask"]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.any(mask & ~border):
            continue
        missing = border & ~mask
        if not (2 <= int(missing.sum()) <= 4):
            continue
        if len(_mask_components(missing)) != 1:
            continue
        frames.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "mask": mask,
                "gap_mask": missing,
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _extract_key(grid: np.ndarray, frame: Dict[str, object], bg: int) -> Dict[str, object]:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    comps = _monochrome_components(inner, bg)
    assert len(comps) == 1, "each frame must contain exactly one key component"
    key = comps[0]
    coords = key["coords"] + np.array([r0 + 1, c0 + 1], dtype=int)
    bbox = _bbox_from_coords(coords)
    return {
        "color": int(key["color"]),
        "coords": coords,
        "bbox": bbox,
        "mask": _coords_to_mask(coords, bbox),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's inner key to an outside stamp, erase the key and stamp,
    and fill the frame's missing border piece with the frame color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one frame"

    frame_mask = np.zeros(g.shape, dtype=bool)
    key_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        coords = frame["coords"]
        frame_mask[coords[:, 0], coords[:, 1]] = True

    for frame in frames:
        key = _extract_key(g, frame, bg)
        frame["key"] = key
        assert _shape_key(frame["gap_mask"]) == _shape_key(key["mask"])
        coords = key["coords"]
        key_mask[coords[:, 0], coords[:, 1]] = True

    candidates: List[Dict[str, object]] = []
    for comp in _monochrome_components(g, bg):
        coords = comp["coords"]
        if np.all(frame_mask[coords[:, 0], coords[:, 1]]):
            continue
        if np.all(key_mask[coords[:, 0], coords[:, 1]]):
            continue
        candidates.append(comp)

    by_shape: Dict[Tuple[int, int, bytes], List[int]] = {}
    for idx, comp in enumerate(candidates):
        by_shape.setdefault(_shape_key(comp["mask"]), []).append(idx)

    out = g.copy()
    used_candidates: set[int] = set()
    for frame in frames:
        target_shape = _shape_key(frame["key"]["mask"])
        options = [idx for idx in by_shape.get(target_shape, []) if idx not in used_candidates]
        assert len(options) == 1, "each key must have one matching outside stamp"
        cand = candidates[options[0]]
        used_candidates.add(options[0])

        key_coords = frame["key"]["coords"]
        out[key_coords[:, 0], key_coords[:, 1]] = bg

        cand_coords = cand["coords"]
        out[cand_coords[:, 0], cand_coords[:, 1]] = bg

        r0, c0, _r1, _c1 = frame["bbox"]
        gap_coords = np.argwhere(frame["gap_mask"]) + np.array([r0, c0], dtype=int)
        out[gap_coords[:, 0], gap_coords[:, 1]] = int(frame["color"])

    return out


BASE_SHAPES: Dict[str, np.ndarray] = {
    "I2": np.array([[1, 1]], dtype=bool),
    "I3": np.array([[1, 1, 1]], dtype=bool),
    "L3": np.array([[1, 1], [1, 0]], dtype=bool),
    "I4": np.array([[1, 1, 1, 1]], dtype=bool),
    "L4": np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
}

FRAME_COUNT_CHOICES = [1, 2, 3, 4]
GRID_SIDE_MIN = {1: 18, 2: 22, 3: 25, 4: 28}
GRID_SIDE_MAX = 34
FRAME_SIDE_RANGE = (6, 11)
PAD_FRAME = 2
PAD_STAMP = 1
GEN_ATTEMPTS = 500


def _reserve_ok(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    hh, ww = reserved.shape
    if r < 0 or c < 0 or r + h > hh or c + w > ww:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(hh, r + h + pad)
    cc1 = min(ww, c + w + pad)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _reserve_mark(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    hh, ww = reserved.shape
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(hh, r + h + pad)
    cc1 = min(ww, c + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r: int, c: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords + np.array([r, c], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    return coords


def _sample_palette() -> Tuple[int, int, int, int]:
    colors = list(range(1, 10))
    chosen = RNG.choice(colors, size=3, replace=False)
    return 0, int(chosen[0]), int(chosen[1]), int(chosen[2])


def _oriented_gap(shape_name: str, outer_h: int, outer_w: int) -> Tuple[np.ndarray, str]:
    base = BASE_SHAPES[shape_name]
    if shape_name.startswith("I"):
        length = int(base.sum())
        side = random.choice(["top", "bottom", "left", "right"])
        gap = np.zeros((outer_h, outer_w), dtype=bool)
        if side in ("top", "bottom"):
            start = int(RNG.integers(1, outer_w - length))
            row = 0 if side == "top" else outer_h - 1
            gap[row, start : start + length] = True
        else:
            start = int(RNG.integers(1, outer_h - length))
            col = 0 if side == "left" else outer_w - 1
            gap[start : start + length, col] = True
        return gap, side

    corner_rotations = [
        ("tl", 0, lambda h, w, ph, pw: (0, 0)),
        ("bl", 1, lambda h, w, ph, pw: (h - ph, 0)),
        ("br", 2, lambda h, w, ph, pw: (h - ph, w - pw)),
        ("tr", 3, lambda h, w, ph, pw: (0, w - pw)),
    ]
    corner, rot, origin_fn = random.choice(corner_rotations)
    patch = np.rot90(base, rot)
    ph, pw = patch.shape
    r0, c0 = origin_fn(outer_h, outer_w, ph, pw)
    gap = np.zeros((outer_h, outer_w), dtype=bool)
    gap[r0 : r0 + ph, c0 : c0 + pw] = patch
    return gap, corner


def _random_key(shape_name: str, inner_h: int, inner_w: int) -> np.ndarray:
    variants = _rotation_variants(BASE_SHAPES[shape_name])
    mask = variants[int(RNG.integers(0, len(variants)))]
    kh, kw = mask.shape
    top = int(RNG.integers(0, inner_h - kh + 1))
    left = int(RNG.integers(0, inner_w - kw + 1))
    patch = np.zeros((inner_h, inner_w), dtype=bool)
    patch[top : top + kh, left : left + kw] = mask
    return patch


def _place_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    shape_name: str,
    frame_color: int,
    key_color: int,
) -> Optional[Dict[str, object]]:
    min_side = max(FRAME_SIDE_RANGE[0], max(BASE_SHAPES[shape_name].shape) + 3)
    max_side = FRAME_SIDE_RANGE[1]
    outer_h = int(RNG.integers(min_side, max_side + 1))
    outer_w = int(RNG.integers(min_side, max_side + 1))
    gap_mask, gap_kind = _oriented_gap(shape_name, outer_h, outer_w)

    border = np.zeros((outer_h, outer_w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    frame_mask = border & ~gap_mask
    key_patch = _random_key(shape_name, outer_h - 2, outer_w - 2)

    h, w = grid.shape
    for _ in range(120):
        r = int(RNG.integers(2, h - outer_h - 1))
        c = int(RNG.integers(2, w - outer_w - 1))
        if not _reserve_ok(reserved, r, c, outer_h, outer_w, PAD_FRAME):
            continue
        frame_coords = _paste_mask(grid, frame_mask, r, c, frame_color)
        key_coords = _paste_mask(grid, key_patch, r + 1, c + 1, key_color)
        _reserve_mark(reserved, r, c, outer_h, outer_w, PAD_FRAME)
        gap_coords = np.argwhere(gap_mask) + np.array([r, c], dtype=int)
        return {
            "shape_name": shape_name,
            "gap_kind": gap_kind,
            "bbox": (r, c, r + outer_h - 1, c + outer_w - 1),
            "gap_coords": gap_coords,
            "frame_coords": frame_coords,
            "key_coords": key_coords,
        }
    return None


def _random_candidate_patch(shape_name: str) -> np.ndarray:
    variants = _rotation_variants(BASE_SHAPES[shape_name])
    return variants[int(RNG.integers(0, len(variants)))]


def _place_edge_stamp(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    color: int,
) -> Optional[np.ndarray]:
    h, w = grid.shape
    ph, pw = patch.shape
    edges = ["top", "bottom", "left", "right"]
    for _ in range(120):
        edge = random.choice(edges)
        if edge == "top":
            r = 0
            c = int(RNG.integers(0, w - pw + 1))
        elif edge == "bottom":
            r = h - ph
            c = int(RNG.integers(0, w - pw + 1))
        elif edge == "left":
            r = int(RNG.integers(0, h - ph + 1))
            c = 0
        else:
            r = int(RNG.integers(0, h - ph + 1))
            c = w - pw
        if not _reserve_ok(reserved, r, c, ph, pw, PAD_STAMP):
            continue
        coords = _paste_mask(grid, patch, r, c, color)
        _reserve_mark(reserved, r, c, ph, pw, PAD_STAMP)
        return coords
    return None


def _expected_output(inp: np.ndarray, frames: List[Dict[str, object]], used_stamps: List[np.ndarray], frame_color: int) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        key_coords = frame["key_coords"]
        out[key_coords[:, 0], key_coords[:, 1]] = 0
        gap_coords = frame["gap_coords"]
        out[gap_coords[:, 0], gap_coords[:, 1]] = frame_color
    for coords in used_stamps:
        out[coords[:, 0], coords[:, 1]] = 0
    return out


def _gen_attempt() -> Optional[np.ndarray]:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    grid_min = GRID_SIDE_MIN[frame_count]
    height = int(RNG.integers(grid_min, GRID_SIDE_MAX + 1))
    width = int(RNG.integers(grid_min, GRID_SIDE_MAX + 1))

    bg, frame_color, key_color, stamp_color = _sample_palette()
    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    shape_names = list(BASE_SHAPES)
    RNG.shuffle(shape_names)
    used_shape_names = shape_names[:frame_count]
    free_shape_names = shape_names[frame_count:]

    frames: List[Dict[str, object]] = []
    for shape_name in used_shape_names:
        frame = _place_frame(grid, reserved, shape_name, frame_color, key_color)
        if frame is None:
            return None
        frames.append(frame)

    used_stamps: List[np.ndarray] = []
    for shape_name in used_shape_names:
        patch = _random_candidate_patch(shape_name)
        coords = _place_edge_stamp(grid, reserved, patch, stamp_color)
        if coords is None:
            return None
        used_stamps.append(coords)

    distractor_count = 0
    if free_shape_names:
        distractor_count = int(RNG.integers(0, min(2, len(free_shape_names)) + 1))
    RNG.shuffle(free_shape_names)
    for shape_name in free_shape_names[:distractor_count]:
        patch = _random_candidate_patch(shape_name)
        if _place_edge_stamp(grid, reserved, patch, stamp_color) is None:
            return None

    expected = _expected_output(grid, frames, used_stamps, frame_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
