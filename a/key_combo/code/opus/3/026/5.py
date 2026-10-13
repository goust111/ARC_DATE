import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PATTERN_COLORS = (BLUE, RED, GREEN, YELLOW)

FRAME_COUNT_RANGE = (4, 6)
GRID_SIDE_RANGE = (18, 24)
MAX_GENERATION_ATTEMPTS = 400

SYMMETRIC_DIMS = ((1, 3), (3, 1), (3, 3))
EVEN_AXIS_DIMS = ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2))


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox of a boolean mask using ARC's (r, c) convention."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Inclusive bbox crop."""
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Connected components of a boolean mask as coordinate arrays."""
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")

    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in steps:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rounded_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return mask
    mask[0, 1 : w - 1] = True
    mask[h - 1, 1 : w - 1] = True
    mask[1 : h - 1, 0] = True
    mask[1 : h - 1, w - 1] = True
    return mask


def _is_rot180_symmetric(sprite: np.ndarray) -> bool:
    return bool(np.array_equal(sprite, np.rot90(sprite, 2)))


def _normalize_to_3x3(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    assert 1 <= h <= 3 and 1 <= w <= 3
    out = np.zeros((3, 3), dtype=int)
    r0 = (3 - h) // 2
    c0 = (3 - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = crop
    return out


def _frame_records(grid: np.ndarray) -> list[dict]:
    grey_components = components_mask(grid == GREY, connectivity=8)
    assert grey_components, "expected at least one rounded frame"

    records = []
    for coords in grey_components:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        box = (r0, c0, r1, c1)
        sub = crop_bbox(grid, box)
        h, w = sub.shape
        expected = _rounded_frame_mask(h, w)
        if not np.array_equal(sub == GREY, expected):
            raise AssertionError("grey component is not a rounded frame")
        assert sub[0, 0] == BLACK and sub[0, -1] == BLACK
        assert sub[-1, 0] == BLACK and sub[-1, -1] == BLACK

        interior = sub[1:-1, 1:-1]
        fg_mask = (interior != BLACK) & (interior != GREY)
        if not fg_mask.any():
            raise AssertionError("frame interior must contain a sprite")
        assert set(map(int, np.unique(interior[fg_mask]))) <= set(PATTERN_COLORS)

        crop_box = bbox(fg_mask)
        assert crop_box is not None
        crop = crop_bbox(interior, crop_box)
        normalized = _normalize_to_3x3(crop)
        records.append(
            {
                "bbox": box,
                "interior_shape": interior.shape,
                "crop_shape": crop.shape,
                "crop": crop,
                "normalized": normalized,
                "raw_symmetric": _is_rot180_symmetric(crop),
                "normalized_symmetric": _is_rot180_symmetric(normalized),
                "has_padding": crop.shape != interior.shape,
            }
        )

    records.sort(key=lambda rec: (rec["bbox"][1], rec["bbox"][0]))
    return records


def _build_output(records: list[dict]) -> np.ndarray:
    symmetric = [rec["normalized"] for rec in records if rec["normalized_symmetric"]]
    asymmetric = [rec["normalized"] for rec in records if not rec["normalized_symmetric"]]
    width = 3 * max(len(symmetric), len(asymmetric))
    assert width > 0

    out = np.zeros((7, width), dtype=int)
    out[3, :] = GREY
    for i, sprite in enumerate(symmetric):
        out[0:3, 3 * i : 3 * (i + 1)] = sprite
    for i, sprite in enumerate(asymmetric):
        out[4:7, 3 * i : 3 * (i + 1)] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract framed sprites, normalize them to 3x3, and stamp them into symmetry-based output bands.

    Input structure
    - The grid is a black background containing several grey rounded rectangular frames.
    - Each frame has black corners, grey outer-border pixels on the non-corner perimeter, and a colored sprite inside.
    - Sprite colors are non-black, non-grey pixels whose cropped bbox ranges from small rectangles up to 3x3.

    Rule
    - Detect every rounded frame and take its interior region.
    - Crop the interior sprite to the bbox of its non-black pixels, then center that crop on a 3x3 black canvas with odd slack biased to the upper-left.
    - Test each normalized 3x3 sprite for 180-degree rotational symmetry.
    - Sort frames by the original frame's left column, place symmetric sprites left-to-right in the top 3 rows, place non-symmetric sprites left-to-right in the bottom 3 rows, and keep a full grey separator row between them.

    Key insight
    - Symmetry is decided only after all source sprites are normalized to the same 3x3 reference frame, so padding introduced by centering can change the classification.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    records = _frame_records(g)
    return _build_output(records)


def _bbox_is_full(arr: np.ndarray) -> bool:
    box = bbox(arr != BLACK)
    if box is None:
        return False
    return box == (0, 0, arr.shape[0] - 1, arr.shape[1] - 1)


def _rot180_orbits(h: int, w: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            orbit = tuple(sorted({(r, c), (h - 1 - r, w - 1 - c)}))
            if orbit not in seen:
                seen.add(orbit)
                out.append(list(orbit))
    return out


def _colorize_orbits(rng: np.random.Generator, arr: np.ndarray, chosen: list[list[tuple[int, int]]]) -> np.ndarray:
    palette = list(PATTERN_COLORS)
    if len(chosen) > 1 and rng.random() < 0.6:
        rng.shuffle(palette)
    for i, orbit in enumerate(chosen):
        color = int(palette[i % len(palette)]) if rng.random() < 0.6 else int(rng.choice(PATTERN_COLORS))
        for r, c in orbit:
            arr[r, c] = color
    return arr


def _build_symmetric_crop(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    orbits = _rot180_orbits(h, w)
    max_cells = min(6, h * w)
    for _ in range(200):
        arr = np.zeros((h, w), dtype=int)
        order = list(range(len(orbits)))
        rng.shuffle(order)
        chosen: list[list[tuple[int, int]]] = []
        used_cells = 0
        for idx in order:
            orbit = orbits[idx]
            if used_cells + len(orbit) > max_cells:
                continue
            if rng.random() < 0.55 or used_cells < 2:
                chosen.append(orbit)
                used_cells += len(orbit)
        if not (2 <= used_cells <= 6):
            continue
        arr = _colorize_orbits(rng, arr, chosen)
        if _bbox_is_full(arr):
            return arr
    raise RuntimeError(f"failed to build symmetric {h}x{w} crop")


def _build_asymmetric_crop(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    max_cells = min(6, h * w)
    for _ in range(400):
        arr = np.zeros((h, w), dtype=int)
        count = int(rng.integers(2, max_cells + 1))
        chosen_idx = rng.choice(len(cells), size=count, replace=False)
        for idx in chosen_idx:
            r, c = cells[int(idx)]
            arr[r, c] = int(rng.choice(PATTERN_COLORS))
        if not _bbox_is_full(arr):
            continue
        if not _is_rot180_symmetric(arr):
            return arr
    raise RuntimeError(f"failed to build asymmetric {h}x{w} crop")


def _sample_crop(rng: np.random.Generator, mode: str) -> np.ndarray:
    for _ in range(200):
        if mode == "sym":
            h, w = SYMMETRIC_DIMS[int(rng.integers(0, len(SYMMETRIC_DIMS)))]
            crop = _build_symmetric_crop(rng, h, w)
            if _is_rot180_symmetric(_normalize_to_3x3(crop)):
                return crop
        elif mode == "asym_even_counterexample":
            h, w = EVEN_AXIS_DIMS[int(rng.integers(0, len(EVEN_AXIS_DIMS)))]
            crop = _build_symmetric_crop(rng, h, w)
            if not _is_rot180_symmetric(_normalize_to_3x3(crop)):
                return crop
        elif mode == "asym_odd":
            h, w = SYMMETRIC_DIMS[int(rng.integers(0, len(SYMMETRIC_DIMS)))]
            crop = _build_asymmetric_crop(rng, h, w)
            if not _is_rot180_symmetric(_normalize_to_3x3(crop)):
                return crop
        elif mode == "asym_even":
            h, w = EVEN_AXIS_DIMS[int(rng.integers(0, len(EVEN_AXIS_DIMS)))]
            crop = _build_asymmetric_crop(rng, h, w)
            if not _is_rot180_symmetric(_normalize_to_3x3(crop)):
                return crop
        else:
            raise ValueError(f"unknown mode: {mode}")
    raise RuntimeError(f"failed to sample crop for mode {mode}")


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _draw_frame(grid: np.ndarray, r0: int, c0: int, interior: np.ndarray) -> None:
    ih, iw = interior.shape
    h = ih + 2
    w = iw + 2
    mask = _rounded_frame_mask(h, w)
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = GREY
    sub[1:-1, 1:-1] = interior


def _place_frames(rng: np.random.Generator, frames: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    used_cols: set[int] = set()

    order = sorted(range(len(frames)), key=lambda i: frames[i]["outer_h"] * frames[i]["outer_w"], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        spec = frames[idx]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - oh + 1):
            for c0 in range(w - ow + 1):
                if c0 in used_cols:
                    continue
                box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                if all(_boxes_separated(box, other, margin=1) for other in placed_boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        placements[idx] = (r0, c0)
        placed_boxes.append((r0, c0, r0 + oh - 1, c0 + ow - 1))
        used_cols.add(c0)

    for idx, spec in enumerate(frames):
        r0, c0 = placements[idx]
        interior = np.zeros((spec["interior_h"], spec["interior_w"]), dtype=int)
        pr = spec["pad_top"]
        pc = spec["pad_left"]
        ch, cw = spec["crop"].shape
        interior[pr : pr + ch, pc : pc + cw] = spec["crop"]
        _draw_frame(grid, r0, c0, interior)
    return grid


def _build_frame_specs(rng: np.random.Generator, n_frames: int) -> list[dict]:
    modes = ["sym", "asym_even_counterexample", "asym_odd"]
    extra_modes = ["sym", "sym", "asym_even_counterexample", "asym_odd", "asym_even"]
    while len(modes) < n_frames:
        modes.append(extra_modes[int(rng.integers(0, len(extra_modes)))])
    rng.shuffle(modes)

    padded_index = int(rng.integers(0, n_frames))
    specs = []
    for i, mode in enumerate(modes):
        crop = _sample_crop(rng, mode)
        ch, cw = crop.shape
        pads = rng.integers(0, 2, size=4)
        if i == padded_index and not np.any(pads):
            pads[int(rng.integers(0, 4))] = 1
        pad_top, pad_bottom, pad_left, pad_right = map(int, pads)
        specs.append(
            {
                "mode": mode,
                "crop": crop,
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
                "interior_h": ch + pad_top + pad_bottom,
                "interior_w": cw + pad_left + pad_right,
                "outer_h": ch + pad_top + pad_bottom + 2,
                "outer_w": cw + pad_left + pad_right + 2,
            }
        )
    return specs


def _validate_generated(grid: np.ndarray) -> None:
    records = _frame_records(grid)
    assert FRAME_COUNT_RANGE[0] <= len(records) <= FRAME_COUNT_RANGE[1]
    cols = [rec["bbox"][1] for rec in records]
    assert len(cols) == len(set(cols)), "frame sorting columns must be unique"
    assert any(rec["normalized_symmetric"] for rec in records)
    assert any(not rec["normalized_symmetric"] for rec in records)
    assert any(rec["raw_symmetric"] and not rec["normalized_symmetric"] for rec in records)
    assert any(rec["crop_shape"] in SYMMETRIC_DIMS and not rec["normalized_symmetric"] for rec in records)
    assert any(rec["has_padding"] for rec in records)
    assert len({rec["normalized"].tobytes() for rec in records}) >= 2

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            assert _boxes_separated(records[i]["bbox"], records[j]["bbox"], margin=1)

    expected = _build_output(records)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert actual.shape[0] == 7 and actual.shape[1] <= 30


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1] + 1))
    specs = _build_frame_specs(rng, n_frames)

    min_h = max(GRID_SIDE_RANGE[0], max(spec["outer_h"] for spec in specs) + 2)
    min_w = max(GRID_SIDE_RANGE[0], max(spec["outer_w"] for spec in specs) + 2)
    max_h = GRID_SIDE_RANGE[1]
    max_w = GRID_SIDE_RANGE[1]
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(40):
        h = int(rng.integers(min_h, max_h + 1))
        w = int(rng.integers(min_w, max_w + 1))
        grid = _place_frames(rng, specs, h, w)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
