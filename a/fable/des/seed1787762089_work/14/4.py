from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7)
TRANSFORMS = ("cw", "ccw", "hflip", "vflip")
SPRITE_SIDES = (3, 4)
FRAME_CORNERS = ("tl", "tr", "bl", "br")
GENERATION_RETRIES = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _component_infos(mask: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for coords in _components(mask):
        bbox, tight = _tight_mask(coords)
        infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": tight,
                "area": int(len(coords)),
            }
        )
    return infos


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "cw":
        return np.rot90(mask, 3)
    if name == "ccw":
        return np.rot90(mask, 1)
    if name == "hflip":
        return np.fliplr(mask)
    if name == "vflip":
        return np.flipud(mask)
    raise ValueError(f"unknown transform: {name}")


def _transform_map(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: _apply_transform(mask, name) for name in TRANSFORMS}


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _find_legend(grid: np.ndarray) -> dict[str, object]:
    gray_infos = _component_infos(grid == GRAY)
    candidates: list[dict[str, object]] = []
    for i, frame in enumerate(gray_infos):
        frame_mask = frame["mask"]
        if not _is_rectangular_frame(frame_mask):
            continue
        r0, c0, r1, c1 = frame["bbox"]
        inner: list[dict[str, object]] = []
        for j, other in enumerate(gray_infos):
            if i == j:
                continue
            rr0, cc0, rr1, cc1 = other["bbox"]
            if r0 < rr0 and rr1 < r1 and c0 < cc0 and cc1 < c1:
                inner.append(other)
        if len(inner) != 2:
            continue
        inner.sort(key=lambda info: (info["bbox"][1], info["bbox"][0]))
        candidates.append({"frame": frame, "inner": inner})
    assert len(candidates) == 1, "expected exactly one legend frame"
    return candidates[0]


def _infer_demo_transform(src: np.ndarray, dst: np.ndarray) -> str:
    matches = [name for name, transformed in _transform_map(src).items() if np.array_equal(transformed, dst)]
    assert len(matches) == 1, "legend transform must be uniquely determined"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend = _find_legend(g)
    frame = legend["frame"]
    inner = legend["inner"]

    src_mask = inner[0]["mask"]
    dst_mask = inner[1]["mask"]
    assert src_mask.shape == dst_mask.shape
    assert src_mask.shape[0] == src_mask.shape[1]

    transform_name = _infer_demo_transform(src_mask, dst_mask)
    target_mask = _apply_transform(src_mask, transform_name)

    out = g.copy()
    for info in [frame, *inner]:
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GRAY):
            continue
        for info in _component_infos(g == color):
            bbox = info["bbox"]
            mask = info["mask"]
            if mask.shape != src_mask.shape or not np.array_equal(mask, src_mask):
                continue
            r0, c0, r1, c1 = bbox
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[:] = BLACK
            patch[target_mask] = color

    return out


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    h, w = mask.shape
    return int(h), int(w), mask.astype(np.uint8).tobytes()


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_unambiguous_demo(mask: np.ndarray) -> bool:
    keys = {_mask_key(transformed) for transformed in _transform_map(mask).values()}
    return len(keys) == len(TRANSFORMS)


def _random_connected_mask(
    side: int,
    banned: set[tuple[int, int, bytes]] | None = None,
    require_unambiguous: bool = False,
) -> np.ndarray | None:
    banned = banned or set()
    min_cells = max(4, side + 1)
    max_cells = side * side - 1
    for _ in range(250):
        target = random.randint(min_cells, max_cells)
        occupied = {(random.randrange(side), random.randrange(side))}
        while len(occupied) < target:
            frontier = []
            seen = set()
            for r, c in occupied:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    cell = (nr, nc)
                    if 0 <= nr < side and 0 <= nc < side and cell not in occupied and cell not in seen:
                        seen.add(cell)
                        frontier.append(cell)
            if not frontier:
                break
            occupied.add(random.choice(frontier))

        mask = np.zeros((side, side), dtype=bool)
        for r, c in occupied:
            mask[r, c] = True
        if not _touches_all_sides(mask):
            continue
        key = _mask_key(mask)
        if key in banned:
            continue
        if require_unambiguous and not _is_unambiguous_demo(mask):
            continue
        return mask
    return None


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _candidate_positions(blocked: np.ndarray, side: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - side + 1):
        for c0 in range(w - side + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + side + 1)
            cc1 = min(w, c0 + side + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _render_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _legend_origin(height: int, width: int, frame_h: int, frame_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - frame_w
    if corner == "bl":
        return height - frame_h, 0
    if corner == "br":
        return height - frame_h, width - frame_w
    raise ValueError(f"unknown corner: {corner}")


def _build_expected(grid: np.ndarray, legend_mask: np.ndarray, placements: list[dict[str, object]], target_mask: np.ndarray) -> np.ndarray:
    expected = grid.copy()
    expected[legend_mask] = BLACK
    for obj in placements:
        if not obj["match"]:
            continue
        side = int(obj["side"])
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        color = int(obj["color"])
        patch = expected[r0 : r0 + side, c0 : c0 + side]
        patch[:] = BLACK
        patch[target_mask] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SPRITE_SIDES)
    src_mask = _random_connected_mask(side, require_unambiguous=True)
    if src_mask is None:
        return None

    transform_name = random.choice(TRANSFORMS)
    target_mask = _apply_transform(src_mask, transform_name)
    gap = random.choice((1, 2))
    frame_h = side + 4
    frame_w = 2 * side + gap + 4

    match_count = random.choice((2, 3))
    distractor_count = random.choice((2, 3))
    total_count = match_count + distractor_count
    colors = random.sample(OBJECT_COLORS, total_count)

    used_distractor_keys = {_mask_key(src_mask)}
    distractor_specs: list[tuple[int, np.ndarray]] = []
    transformed_variants = [
        transformed
        for transformed in _transform_map(src_mask).values()
        if not np.array_equal(transformed, src_mask)
    ]
    if distractor_count > 0 and random.random() < 0.75:
        variant = random.choice(transformed_variants)
        distractor_specs.append((side, variant))
        used_distractor_keys.add(_mask_key(variant))

    while len(distractor_specs) < distractor_count:
        distractor_side = side if len(distractor_specs) == 0 else random.choice(SPRITE_SIDES)
        mask = _random_connected_mask(distractor_side, banned=used_distractor_keys, require_unambiguous=False)
        if mask is None:
            return None
        distractor_specs.append((distractor_side, mask))
        used_distractor_keys.add(_mask_key(mask))

    objects: list[dict[str, object]] = []
    for idx in range(match_count):
        objects.append({"side": side, "mask": src_mask, "color": colors[idx], "match": True})
    for idx, (distractor_side, mask) in enumerate(distractor_specs):
        objects.append(
            {
                "side": distractor_side,
                "mask": mask,
                "color": colors[match_count + idx],
                "match": False,
            }
        )
    random.shuffle(objects)

    min_h = max(18, frame_h + 6)
    min_w = max(20, frame_w + 6)
    height = random.randint(min_h, 26)
    width = random.randint(min_w, 26)
    corner = random.choice(FRAME_CORNERS)
    frame_r0, frame_c0 = _legend_origin(height, width, frame_h, frame_w, corner)

    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    _mark_blocked(blocked, frame_r0, frame_c0, frame_h, frame_w, pad=1)

    placements: list[dict[str, object]] = []
    for obj in sorted(objects, key=lambda item: int(item["side"]), reverse=True):
        obj_side = int(obj["side"])
        positions = _candidate_positions(blocked, obj_side)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _render_mask(grid, r0, c0, obj["mask"], int(obj["color"]))
        _mark_blocked(blocked, r0, c0, obj_side, obj_side, pad=1)
        placements.append({**obj, "r0": r0, "c0": c0})

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = False

    grid[frame_r0, frame_c0 : frame_c0 + frame_w] = GRAY
    grid[frame_r0 + frame_h - 1, frame_c0 : frame_c0 + frame_w] = GRAY
    grid[frame_r0 : frame_r0 + frame_h, frame_c0] = GRAY
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 + frame_w - 1] = GRAY
    legend_mask[frame_r0, frame_c0 : frame_c0 + frame_w] = True
    legend_mask[frame_r0 + frame_h - 1, frame_c0 : frame_c0 + frame_w] = True
    legend_mask[frame_r0 : frame_r0 + frame_h, frame_c0] = True
    legend_mask[frame_r0 : frame_r0 + frame_h, frame_c0 + frame_w - 1] = True

    left_r0 = frame_r0 + 2
    left_c0 = frame_c0 + 2
    right_r0 = left_r0
    right_c0 = left_c0 + side + gap
    _render_mask(grid, left_r0, left_c0, src_mask, GRAY)
    _render_mask(grid, right_r0, right_c0, target_mask, GRAY)
    legend_mask[left_r0 : left_r0 + side, left_c0 : left_c0 + side] |= src_mask
    legend_mask[right_r0 : right_r0 + side, right_c0 : right_c0 + side] |= target_mask

    expected = _build_expected(grid, legend_mask, placements, target_mask)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
