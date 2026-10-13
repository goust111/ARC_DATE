import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GRID_SIZE = 30
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _all_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps.extend(_components_of_color(grid, color))
    return comps


def _frame_info(component: dict) -> dict | None:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    pts = {tuple(map(int, rc)) for rc in component["coords"]}
    boundary = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0, r1 + 1):
        boundary.add((r, c0))
        boundary.add((r, c1))

    if not pts.issubset(boundary):
        return None

    missing = list(boundary - pts)
    if len(missing) != 1:
        return None

    gap_r, gap_c = missing[0]
    if gap_r in (r0, r1) and gap_c in (c0, c1):
        return None

    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    elif gap_c == c1:
        side = "right"
    else:
        return None

    if component["area"] != len(boundary) - 1:
        return None

    info = dict(component)
    info["gap"] = (int(gap_r), int(gap_c))
    info["side"] = side
    return info


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray, tuple[int, int], np.ndarray]:
    bg = get_bg(grid)
    frame_candidates = []
    for component in _all_components(grid, bg):
        info = _frame_info(component)
        if info is not None:
            frame_candidates.append(info)

    assert frame_candidates, "expected at least one gapped frame"
    frames = sorted(frame_candidates, key=lambda comp: _bbox_area(comp["bbox"]))
    for inner, outer in zip(frames, frames[1:]):
        assert _contains(outer["bbox"], inner["bbox"]), "frames must be properly nested"

    frame_mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        coords = frame["coords"]
        frame_mask[coords[:, 0], coords[:, 1]] = True

    payload_coords = np.argwhere((grid != bg) & (~frame_mask))
    assert payload_coords.size > 0, "expected a payload inside the innermost frame"

    pr0, pc0 = payload_coords.min(axis=0)
    pr1, pc1 = payload_coords.max(axis=0)
    patch = grid[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert np.any(patch != bg)
    return bg, frames, patch, (int(pr0), int(pc0)), payload_coords


def _outside_position(frame: dict, patch_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    gap_r, gap_c = frame["gap"]
    h, w = patch_shape

    if frame["side"] == "top":
        return int(r0 - h), int(gap_c - w // 2)
    if frame["side"] == "bottom":
        return int(r1 + 1), int(gap_c - w // 2)
    if frame["side"] == "left":
        return int(gap_r - h // 2), int(c0 - w)
    return int(gap_r - h // 2), int(c1 + 1)


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int) -> None:
    h, w = patch.shape
    mask = patch != transparent
    target = grid[top : top + h, left : left + w]
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the gap direction of each nested frame and eject the inner payload outside
    the outermost frame.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, patch, _payload_pos, payload_coords = _extract_scene(g)

    out = g.copy()
    out[payload_coords[:, 0], payload_coords[:, 1]] = bg

    final_top = final_left = None
    for frame in frames:
        final_top, final_left = _outside_position(frame, patch.shape)

    assert final_top is not None and final_left is not None
    h, w = patch.shape
    assert 0 <= final_top <= g.shape[0] - h
    assert 0 <= final_left <= g.shape[1] - w
    _paste(out, patch, final_top, final_left, bg)
    return out


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = BG


def _random_connected_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    min_area = max(1, (h * w + 1) // 2)
    for _ in range(100):
        target = int(rng.integers(min_area, h * w + 1))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        chosen = {start}
        while len(chosen) < target:
            frontier = []
            frontier_set = set()
            for r, c in chosen:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    pos = (nr, nc)
                    if 0 <= nr < h and 0 <= nc < w and pos not in chosen and pos not in frontier_set:
                        frontier.append(pos)
                        frontier_set.add(pos)
            if not frontier:
                break
            picked = frontier[int(rng.integers(0, len(frontier)))]
            chosen.add(picked)

        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if 0 in rows and (h - 1) in rows and 0 in cols and (w - 1) in cols:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in chosen:
                mask[r, c] = True
            return mask

    return np.ones((h, w), dtype=bool)


def _sample_gap(
    bbox: tuple[int, int, int, int],
    side: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, int(rng.integers(c0 + 1, c1))
    if side == "bottom":
        return r1, int(rng.integers(c0 + 1, c1))
    if side == "left":
        return int(rng.integers(r0 + 1, r1)), c0
    return int(rng.integers(r0 + 1, r1)), c1


def _split_extra(total: int, rng: np.random.Generator) -> tuple[int, int]:
    left = int(rng.integers(0, total + 1))
    return left, total - left


def _make_payload_patch(
    h: int,
    w: int,
    colors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = _random_connected_mask(h, w, rng)
    coords = np.argwhere(mask)
    max_colors = min(3, len(coords), len(colors))
    n_colors = int(rng.integers(1, max_colors + 1))
    chosen = [int(c) for c in colors[:n_colors]]

    labels = np.empty(len(coords), dtype=int)
    for i, color in enumerate(chosen):
        labels[i] = color
    if len(coords) > n_colors:
        labels[n_colors:] = rng.choice(chosen, size=len(coords) - n_colors)
    rng.shuffle(labels)

    patch = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(coords, labels):
        patch[int(r), int(c)] = int(color)
    return patch


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _expected_output(
    grid: np.ndarray,
    patch: np.ndarray,
    payload_top: int,
    payload_left: int,
    frames: list[dict],
) -> np.ndarray:
    out = grid.copy()
    h, w = patch.shape
    payload_mask = patch != BG
    src = out[payload_top : payload_top + h, payload_left : payload_left + w]
    src[payload_mask] = BG

    final_top = final_left = None
    for frame in frames:
        final_top, final_left = _outside_position(frame, patch.shape)

    assert final_top is not None and final_left is not None
    _paste(out, patch, final_top, final_left, BG)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    payload_h = int(rng.integers(1, 4))
    payload_w = int(rng.integers(1, 4))
    n_layers = int(rng.integers(2, 4))

    inner_h = payload_h + 4
    inner_w = payload_w + 4
    step_r = payload_h + 1
    step_c = payload_w + 1

    bboxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(1, n_layers):
        r0, c0, r1, c1 = bboxes[-1]
        bboxes.append((r0 - step_r, c0 - step_c, r1 + step_r, c1 + step_c))

    outer_bbox = bboxes[-1]
    outer_h = outer_bbox[2] - outer_bbox[0] + 1
    outer_w = outer_bbox[3] - outer_bbox[1] + 1

    base_h = outer_h + 2 * payload_h
    base_w = outer_w + 2 * payload_w
    if base_h > MAX_GRID_SIZE or base_w > MAX_GRID_SIZE:
        return None

    extra_h = int(rng.integers(0, min(2, MAX_GRID_SIZE - base_h) + 1))
    extra_w = int(rng.integers(0, min(2, MAX_GRID_SIZE - base_w) + 1))
    top_extra, bottom_extra = _split_extra(extra_h, rng)
    left_extra, right_extra = _split_extra(extra_w, rng)

    top_margin = payload_h + top_extra
    bottom_margin = payload_h + bottom_extra
    left_margin = payload_w + left_extra
    right_margin = payload_w + right_extra

    shift_r = top_margin - outer_bbox[0]
    shift_c = left_margin - outer_bbox[1]
    shifted_bboxes = [_shift_bbox(bbox, shift_r, shift_c) for bbox in bboxes]

    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    palette = rng.permutation(PALETTE)
    frame_colors = [int(c) for c in palette[:n_layers]]
    payload_colors = palette[n_layers:]
    patch = _make_payload_patch(payload_h, payload_w, payload_colors, rng)

    top_space = int(rng.integers(0, 3))
    left_space = int(rng.integers(0, 3))
    inner_bbox = shifted_bboxes[0]
    payload_top = inner_bbox[0] + 1 + top_space
    payload_left = inner_bbox[1] + 1 + left_space

    frames: list[dict] = []
    for bbox, color in zip(shifted_bboxes, frame_colors):
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        gap = _sample_gap(bbox, side, rng)
        _paint_frame(grid, bbox, color, gap)
        frames.append(
            {
                "bbox": bbox,
                "color": int(color),
                "gap": (int(gap[0]), int(gap[1])),
                "side": side,
            }
        )

    _paste(grid, patch, payload_top, payload_left, BG)
    expected = _expected_output(grid, patch, payload_top, payload_left, frames)

    try:
        bg, found_frames, found_patch, found_origin, _payload_coords = _extract_scene(grid)
    except AssertionError:
        return None

    if bg != BG or len(found_frames) != n_layers:
        return None
    if found_patch.shape != patch.shape or not np.array_equal(found_patch, patch):
        return None
    if found_origin != (payload_top, payload_left):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
