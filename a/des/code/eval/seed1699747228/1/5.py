import numpy as np
from collections import deque


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
KEY_COLORS = (1, 2, 3, 4, 6, 8)
GENERATE_MAX_RETRIES = 300


VERTICAL_PATTERNS = (
    np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
)

HORIZONTAL_PATTERNS = (
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]], dtype=bool),
)

ASYMMETRIC_PATTERNS = (
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return _bbox_from_coords(coords)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(mask.shape) + tuple(mask.astype(np.uint8).ravel())


def _symmetry_kind(mask: np.ndarray) -> str:
    if np.array_equal(mask, np.fliplr(mask)):
        return "vertical"
    if np.array_equal(mask, np.flipud(mask)):
        return "horizontal"
    return "none"


def _transform_pattern(mask: np.ndarray) -> np.ndarray:
    kind = _symmetry_kind(mask)
    if kind == "vertical":
        return mask.copy()
    if kind == "horizontal":
        return np.fliplr(mask)
    return np.rot90(mask, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key tooth to the complementary chest lockhole, then recolor and
    transform the chest's inner pattern according to its symmetry type.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_rows = np.where(np.all(g == GREY, axis=1))[0]
    assert len(separator_rows) == 1
    sep = int(separator_rows[0])

    key_components = _components(g[:sep, :] != BG)
    keys = []
    for coords in key_components:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, patch[patch != BG]))
        assert len(colors) == 1
        color = colors.pop()
        mask = patch == color
        assert mask.shape[1] == 2 and mask.shape[0] >= 5
        assert np.all(mask[:2, :])
        tooth = _normalize_mask(mask[2:, :])
        keys.append(
            {
                "color": color,
                "tooth": tooth,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        )

    chest_components = _components(g[sep + 1 :, :] == ORANGE)
    chests = []
    for coords in chest_components:
        rr0, cc0, rr1, cc1 = _bbox_from_coords(coords)
        r0, c0, r1, c1 = rr0 + sep + 1, cc0, rr1 + sep + 1, cc1
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        seal_rows = [r for r in range(1, patch.shape[0] - 1) if np.all(patch[r, :] == ORANGE)]
        assert seal_rows
        seal = int(seal_rows[0])
        lock_mask = patch[:seal, :] != ORANGE
        assert np.any(lock_mask)
        lock = _normalize_mask(lock_mask)

        maroon = patch == MAROON
        assert np.any(maroon)
        pr0, pc0, pr1, pc1 = _bbox_from_mask(maroon)
        pattern = maroon[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

        lock_coords = np.argwhere(lock_mask)
        chests.append(
            {
                "lock": lock,
                "lock_abs": [(int(r0 + r), int(c0 + c)) for r, c in lock_coords],
                "pattern_bbox": (int(r0 + pr0), int(c0 + pc0), int(r0 + pr1), int(c0 + pc1)),
                "pattern": pattern,
            }
        )

    assert len(keys) == len(chests) >= 2
    key_by_shape = {_mask_signature(key["tooth"]): key for key in keys}
    assert len(key_by_shape) == len(keys)

    out = g.copy()
    for key in keys:
        r0, c0, r1, c1 = key["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[key["mask"]] = BG

    for chest in chests:
        sig = _mask_signature(chest["lock"])
        assert sig in key_by_shape
        color = int(key_by_shape[sig]["color"])
        for r, c in chest["lock_abs"]:
            out[r, c] = color

        pr0, pc0, pr1, pc1 = chest["pattern_bbox"]
        transformed = _transform_pattern(chest["pattern"])
        sub = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        sub[:] = BG
        sub[transformed] = color

    return out


def _random_tooth(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(3, 6))
    spine_col = int(rng.integers(0, 2))
    extras = rng.integers(0, 2, size=height, dtype=int)
    if not extras.any():
        extras[int(rng.integers(0, height))] = 1
    mask = np.zeros((height, 2), dtype=bool)
    mask[:, spine_col] = True
    mask[extras.astype(bool), 1 - spine_col] = True
    return mask


def _choose_pattern(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "vertical":
        return VERTICAL_PATTERNS[int(rng.integers(0, len(VERTICAL_PATTERNS)))].copy()
    if kind == "horizontal":
        return HORIZONTAL_PATTERNS[int(rng.integers(0, len(HORIZONTAL_PATTERNS)))].copy()
    return ASYMMETRIC_PATTERNS[int(rng.integers(0, len(ASYMMETRIC_PATTERNS)))].copy()


def _branch_schedule(rng: np.random.Generator, count: int) -> list[str]:
    base = ["vertical", "horizontal", "none"]
    if count == 2:
        choice = rng.choice(base, size=2, replace=False)
        return [str(x) for x in choice]
    order = list(base)
    rng.shuffle(order)
    return order


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pair_count = int(rng.integers(2, 4))
    colors = list(rng.choice(KEY_COLORS, size=pair_count, replace=False))

    teeth: list[np.ndarray] = []
    seen_teeth: set[tuple[int, ...]] = set()
    while len(teeth) < pair_count:
        tooth = _random_tooth(rng)
        sig = _mask_signature(tooth)
        if sig in seen_teeth:
            continue
        seen_teeth.add(sig)
        teeth.append(tooth)

    key_order = list(range(pair_count))
    chest_order = list(range(pair_count))
    rng.shuffle(key_order)
    rng.shuffle(chest_order)

    key_specs = []
    key_x = int(rng.integers(1, 4))
    max_key_bottom = 0
    for idx in key_order:
        tooth = teeth[idx]
        key_mask = np.vstack([np.ones((2, 2), dtype=bool), tooth])
        top = int(rng.integers(1, 4))
        key_specs.append(
            {
                "index": idx,
                "color": colors[idx],
                "mask": key_mask,
                "top": top,
                "left": key_x,
            }
        )
        key_x += key_mask.shape[1] + int(rng.integers(2, 5))
        max_key_bottom = max(max_key_bottom, top + key_mask.shape[0] - 1)

    sep = max_key_bottom + int(rng.integers(1, 3))
    branch_kinds = _branch_schedule(rng, pair_count)

    chest_specs = []
    chest_x = int(rng.integers(1, 4))
    chest_base_top = sep + int(rng.integers(2, 5))
    max_chest_bottom = chest_base_top
    for pos, key_idx in enumerate(chest_order):
        tooth = teeth[key_idx]
        lock_h = tooth.shape[0]
        pattern = _choose_pattern(rng, branch_kinds[pos])
        inner_w = max(int(pattern.shape[1] + rng.integers(2, 4)), 4)
        lower_h = int(pattern.shape[0] + rng.integers(2, 4))
        chest_h = lock_h + lower_h + 2
        chest_w = inner_w + 2
        top = chest_base_top + int(rng.integers(0, 3))
        left = chest_x
        lock_c0 = 1 + (inner_w - 2) // 2
        pat_r = lock_h + 1 + int(rng.integers(0, lower_h - pattern.shape[0] + 1))
        pat_c = 1 + int(rng.integers(0, inner_w - pattern.shape[1] + 1))
        chest_specs.append(
            {
                "key_index": key_idx,
                "top": top,
                "left": left,
                "height": chest_h,
                "width": chest_w,
                "lock_c0": lock_c0,
                "tooth": tooth,
                "pattern": pattern,
                "pattern_pos": (pat_r, pat_c),
            }
        )
        chest_x += chest_w + int(rng.integers(2, 5))
        max_chest_bottom = max(max_chest_bottom, top + chest_h - 1)

    width = max(key_x, chest_x) + int(rng.integers(1, 4))
    height = max_chest_bottom + int(rng.integers(2, 5))
    if width > 40 or height > 30:
        raise ValueError("grid too large")

    grid = np.full((height, width), BG, dtype=int)
    grid[sep, :] = GREY

    for spec in key_specs:
        mask = spec["mask"]
        h, w = mask.shape
        sub = grid[spec["top"] : spec["top"] + h, spec["left"] : spec["left"] + w]
        sub[mask] = int(spec["color"])

    for spec in chest_specs:
        top, left = int(spec["top"]), int(spec["left"])
        h, w = int(spec["height"]), int(spec["width"])
        tooth = spec["tooth"]
        pattern = spec["pattern"]
        pr, pc = spec["pattern_pos"]

        chest = grid[top : top + h, left : left + w]
        chest[: tooth.shape[0] + 1, :] = ORANGE
        chest[:, 0] = ORANGE
        chest[:, -1] = ORANGE
        chest[-1, :] = ORANGE

        for r, c in np.argwhere(tooth):
            chest[int(r), int(spec["lock_c0"] + c)] = BG

        pat_sub = chest[pr : pr + pattern.shape[0], pc : pc + pattern.shape[1]]
        pat_sub[pattern] = MAROON

    expected = grid.copy()
    key_meta = {spec["index"]: spec for spec in key_specs}
    for spec in key_specs:
        mask = spec["mask"]
        h, w = mask.shape
        sub = expected[spec["top"] : spec["top"] + h, spec["left"] : spec["left"] + w]
        sub[mask] = BG

    for spec in chest_specs:
        key = key_meta[int(spec["key_index"])]
        color = int(key["color"])
        top, left = int(spec["top"]), int(spec["left"])
        for r, c in np.argwhere(spec["tooth"]):
            expected[top + int(r), left + int(spec["lock_c0"] + c)] = color

        pattern = spec["pattern"]
        pr, pc = spec["pattern_pos"]
        transformed = _transform_pattern(pattern)
        sub = expected[top + pr : top + pr + pattern.shape[0], left + pc : left + pc + pattern.shape[1]]
        sub[:] = BG
        sub[transformed] = color

    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_MAX_RETRIES):
        try:
            grid, expected = _build_case(rng)
        except ValueError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
