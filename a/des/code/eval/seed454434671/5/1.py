from __future__ import annotations

from collections import defaultdict

import numpy as np


BG = 0
FG_COLORS = tuple(range(1, 10))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

RNG = np.random.default_rng()

# Generator configuration.
PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 5
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 30
MIN_BBOX_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
PARTITION_LIBRARY_TARGET = 12
PARTITION_LIBRARY_ATTEMPTS = 6000
PLACEMENT_ATTEMPTS = 80

_PARTITION_LIBRARY: dict[tuple[int, int], list[np.ndarray]] = {}


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _canonical_partition_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    other = np.logical_not(mask)
    a = mask.astype(np.uint8).tobytes()
    b = other.astype(np.uint8).tobytes()
    return (int(mask.shape[0]), int(mask.shape[1]), a if a < b else b)


def _random_partition_candidate(h: int, w: int) -> np.ndarray:
    prob = float(RNG.uniform(0.28, 0.72))
    return RNG.random((h, w)) < prob


def _valid_partition(mask: np.ndarray) -> bool:
    total = int(mask.size)
    filled = int(mask.sum())
    if filled < 2 or filled > total - 2:
        return False
    other = np.logical_not(mask)
    return (
        _is_connected(mask)
        and _is_connected(other)
        and _spans_bbox(mask)
        and _spans_bbox(other)
    )


def _ensure_partition_library(h: int, w: int) -> None:
    dims = (int(h), int(w))
    masks = _PARTITION_LIBRARY.setdefault(dims, [])
    seen = {_canonical_partition_signature(mask) for mask in masks}
    attempts = 0
    while len(masks) < PARTITION_LIBRARY_TARGET and attempts < PARTITION_LIBRARY_ATTEMPTS:
        attempts += 1
        mask = _random_partition_candidate(h, w)
        if not _valid_partition(mask):
            continue
        sig = _canonical_partition_signature(mask)
        if sig in seen:
            continue
        masks.append(mask.copy())
        seen.add(sig)
    if not masks:
        raise RuntimeError(f"no valid complementary partition found for rectangle {h}x{w}")


def _sample_partition(h: int, w: int) -> np.ndarray:
    _ensure_partition_library(h, w)
    library = _PARTITION_LIBRARY[(int(h), int(w))]
    idx = int(RNG.integers(len(library)))
    return library[idx].copy()


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    fragments = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[int(r) - r0, int(c) - c0] = True
        assert _is_connected(mask), "each color must form exactly one connected fragment"
        fragments.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "mask": mask,
            }
        )
    fragments.sort(key=lambda frag: (frag["r0"], frag["c0"], frag["h"], frag["w"]))
    return fragments


def _pair_fragments(fragments: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, frag in enumerate(fragments):
        groups[(frag["h"], frag["w"])].append(idx)

    pairs: list[dict] = []
    used: set[int] = set()
    for (h, w), indices in groups.items():
        by_mask: dict[bytes, list[int]] = defaultdict(list)
        for idx in indices:
            key = fragments[idx]["mask"].astype(np.uint8).tobytes()
            by_mask[key].append(idx)
        for idx in indices:
            if idx in used:
                continue
            frag = fragments[idx]
            comp_key = np.logical_not(frag["mask"]).astype(np.uint8).tobytes()
            matches = [j for j in by_mask.get(comp_key, []) if j not in used and j != idx]
            assert len(matches) == 1, "each fragment must have a unique complementary mate"
            mate = fragments[matches[0]]
            used.add(idx)
            used.add(matches[0])

            patch = np.full((h, w), int(mate["color"]), dtype=int)
            patch[frag["mask"]] = int(frag["color"])
            pairs.append(
                {
                    "h": h,
                    "w": w,
                    "area": int(h * w),
                    "signature": _canonical_partition_signature(frag["mask"]),
                    "anchor": (
                        min(int(frag["r0"]), int(mate["r0"])),
                        min(int(frag["c0"]), int(mate["c0"])),
                    ),
                    "patch": patch,
                }
            )
    assert len(used) == len(fragments), "all fragments must be paired"
    return pairs


def _stack_rectangles(pairs: list[dict]) -> np.ndarray:
    assert pairs, "at least one complementary pair is required"
    ordered = sorted(
        pairs,
        key=lambda pair: (
            int(pair["area"]),
            int(pair["h"]),
            int(pair["w"]),
            pair["signature"],
            pair["anchor"],
        ),
    )
    out_h = sum(int(pair["h"]) for pair in ordered) + len(ordered) - 1
    out_w = max(int(pair["w"]) for pair in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    r = 0
    for pair in ordered:
        h = int(pair["h"])
        w = int(pair["w"])
        out[r : r + h, :w] = pair["patch"]
        r += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments, rebuild rectangles, then stack them by area."""
    g = np.asarray(grid, dtype=int)
    fragments = _extract_fragments(g)
    assert len(fragments) >= 2 and len(fragments) % 2 == 0
    return _stack_rectangles(_pair_fragments(fragments))


def _pair_sort_key(spec: dict) -> tuple:
    mask = spec["mask"]
    return (
        int(spec["h"] * spec["w"]),
        int(spec["h"]),
        int(spec["w"]),
        _canonical_partition_signature(mask),
    )


def _build_expected_from_specs(pair_specs: list[dict]) -> np.ndarray:
    ordered = sorted(pair_specs, key=_pair_sort_key)
    out_h = sum(int(spec["h"]) for spec in ordered) + len(ordered) - 1
    out_w = max(int(spec["w"]) for spec in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    r = 0
    for spec in ordered:
        h = int(spec["h"])
        w = int(spec["w"])
        patch = np.full((h, w), int(spec["color_b"]), dtype=int)
        patch[spec["mask"]] = int(spec["color_a"])
        out[r : r + h, :w] = patch
        r += h + 1
    return out


def _make_fragment_patches(pair_specs: list[dict]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for spec in pair_specs:
        patch_a = np.full((spec["h"], spec["w"]), BG, dtype=int)
        patch_b = np.full((spec["h"], spec["w"]), BG, dtype=int)
        patch_a[spec["mask"]] = int(spec["color_a"])
        patch_b[np.logical_not(spec["mask"])] = int(spec["color_b"])
        patches.extend([patch_a, patch_b])
    return patches


def _can_place_bbox(blocked: np.ndarray, h: int, w: int, r0: int, c0: int) -> bool:
    H, W = blocked.shape
    rr0 = max(0, r0 - MIN_BBOX_GAP)
    rr1 = min(H, r0 + h + MIN_BBOX_GAP)
    cc0 = max(0, c0 - MIN_BBOX_GAP)
    cc1 = min(W, c0 + w + MIN_BBOX_GAP)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _place_patches(H: int, W: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    order = list(range(len(patches)))
    RNG.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        positions = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if _can_place_bbox(blocked, h, w, r0, c0):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = positions[int(RNG.integers(len(positions)))]
        nz = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][nz] = patch[nz]
        blocked[r0 : r0 + h, c0 : c0 + w] = True
    return grid


def _sample_pair_specs(pair_count: int) -> list[dict] | None:
    dims_pool = [(h, w) for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1) for w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)]
    RNG.shuffle(dims_pool)
    used_areas: set[int] = set()
    selected: list[tuple[int, int, np.ndarray]] = []
    for h, w in dims_pool:
        area = int(h * w)
        if area in used_areas:
            continue
        try:
            mask = _sample_partition(h, w)
        except RuntimeError:
            continue
        selected.append((h, w, mask))
        used_areas.add(area)
        if len(selected) == pair_count:
            break
    if len(selected) != pair_count:
        return None

    colors = list(map(int, RNG.choice(np.array(FG_COLORS, dtype=int), size=2 * pair_count, replace=False)))
    pair_specs = []
    for i, (h, w, base_mask) in enumerate(selected):
        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        mask = base_mask if bool(RNG.integers(0, 2)) else np.logical_not(base_mask)
        pair_specs.append(
            {
                "h": int(h),
                "w": int(w),
                "mask": mask.copy(),
                "color_a": int(color_a),
                "color_b": int(color_b),
            }
        )
    return pair_specs


def _generate_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice(np.array(PAIR_COUNT_CHOICES, dtype=int)))
    pair_specs = _sample_pair_specs(pair_count)
    if pair_specs is None:
        return None

    expected = _build_expected_from_specs(pair_specs)
    patches = _make_fragment_patches(pair_specs)
    footprint_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)
    min_side = max(
        GRID_SIDE_MIN,
        max(int(patch.shape[0]) for patch in patches) + 2,
        max(int(patch.shape[1]) for patch in patches) + 2,
        int(np.ceil(np.sqrt(footprint_area * 2.2))),
    )
    if min_side > GRID_SIDE_MAX:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        H = int(RNG.integers(min_side, GRID_SIDE_MAX + 1))
        W = int(RNG.integers(min_side, GRID_SIDE_MAX + 1))
        grid = _place_patches(H, W, patches)
        if grid is None:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
