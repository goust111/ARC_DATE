import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_GEN_RETRIES = 400
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _color_components(g: np.ndarray, color: int) -> list[np.ndarray]:
    mask = g == int(color)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_crop_non_bg(arr: np.ndarray, bg: int = _BG) -> np.ndarray:
    pts = np.argwhere(arr != int(bg))
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _rotate_cw_tight(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    turned = np.rot90(arr, -int(quarter_turns % 4))
    return _tight_crop_non_bg(turned, _BG)


def _paste_non_bg(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int = _BG) -> None:
    h, w = patch.shape
    assert 0 <= int(top) and 0 <= int(left)
    assert int(top) + h <= out.shape[0]
    assert int(left) + w <= out.shape[1]
    mask = patch != int(bg)
    out[int(top) : int(top) + h, int(left) : int(left) + w][mask] = patch[mask]


def _is_frame_component(coords: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False, None
    if int(len(coords)) != 2 * (h + w) - 4:
        return False, None
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return False, None
    return True, (int(r0), int(c0), int(r1), int(c1))


def _extract_frames(g: np.ndarray, bg: int = _BG) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        for coords in _color_components(g, color):
            ok, bbox = _is_frame_component(coords)
            if not ok or bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if not np.any(inner != int(bg)):
                continue
            stamp = _tight_crop_non_bg(inner, bg)
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "stamp": stamp,
                    "center_x2": int(c0 + c1),
                    "center_y2": int(r0 + r1),
                }
            )
    frames.sort(key=lambda item: (item["center_x2"], item["center_y2"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the stamp inside each colored frame, rotate it by the frame's left-to-right order,
    and stamp it at every external anchor pixel sharing the frame color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _BG
    frames = _extract_frames(g, bg)
    assert frames

    inside_any_frame = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside_any_frame[r0 : r1 + 1, c0 : c1 + 1] = True

    transformed_by_color: dict[int, np.ndarray] = {}
    for i, frame in enumerate(frames):
        transformed_by_color[int(frame["color"])] = _rotate_cw_tight(frame["stamp"], i % 4)

    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or inside_any_frame[r, c]:
                continue
            if color not in transformed_by_color:
                continue
            _paste_non_bg(out, transformed_by_color[color], r, c, bg)

    return out


def _make_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < int(area):
        choices = []
        for r, c in frontier:
            for dr, dc in _DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    choices.append((nr, nc))
        if not choices:
            return _make_connected_mask(h, w, area)
        nr, nc = choices[int(_RNG.integers(len(choices)))]
        mask[nr, nc] = True
        frontier.add((nr, nc))
    return mask


def _make_stamp(required_rotation: int) -> np.ndarray:
    for _ in range(200):
        h = int(_RNG.integers(2, 5))
        w = int(_RNG.integers(2, 5))
        area_lo = max(3, (h * w + 1) // 2)
        area = int(_RNG.integers(area_lo, h * w + 1))
        mask = _make_connected_mask(h, w, area)
        tight_mask = _tight_crop_non_bg(mask.astype(int), 0).astype(bool)
        th, tw = tight_mask.shape
        if th < 2 or tw < 2:
            continue
        cells = np.argwhere(tight_mask)
        color_count = int(min(len(cells), int(_RNG.integers(2, 4))))
        palette = _RNG.choice(np.arange(1, 10), size=color_count, replace=False)
        patch = np.zeros((th, tw), dtype=int)
        order = _RNG.permutation(len(cells))
        for i, idx in enumerate(order[:color_count]):
            r, c = cells[int(idx)]
            patch[int(r), int(c)] = int(palette[i])
        for idx in order[color_count:]:
            r, c = cells[int(idx)]
            patch[int(r), int(c)] = int(palette[int(_RNG.integers(color_count))])
        if len(set(int(v) for v in patch[patch != 0])) < 2:
            continue
        if required_rotation % 4:
            rotated = _rotate_cw_tight(patch, required_rotation)
            if np.array_equal(rotated, patch):
                continue
        return patch
    raise RuntimeError("failed to generate a non-degenerate stamp")


def _sample_centers(width: int, count: int) -> list[int] | None:
    candidates = list(range(5, width - 4))
    _RNG.shuffle(candidates)
    picked = []
    for cx in candidates:
        if all(abs(cx - prev) >= 4 for prev in picked):
            picked.append(int(cx))
            if len(picked) == int(count):
                return sorted(picked)
    return None


def _rect_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(occ.shape[0], int(r0) + int(h) + pad)
    cc1 = min(occ.shape[1], int(c0) + int(w) + pad)
    return not bool(np.any(occ[rr0:rr1, cc0:cc1]))


def _occupy_rect(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(occ.shape[0], int(r0) + int(h) + pad)
    cc1 = min(occ.shape[1], int(c0) + int(w) + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _place_anchor_rect(occ: np.ndarray, patch_h: int, patch_w: int) -> tuple[int, int] | None:
    h, w = occ.shape
    attempts = 250
    for _ in range(attempts):
        r0 = int(_RNG.integers(1, h - patch_h))
        c0 = int(_RNG.integers(1, w - patch_w))
        if _rect_free(occ, r0, c0, patch_h, patch_w, pad=1):
            _occupy_rect(occ, r0, c0, patch_h, patch_w, pad=1)
            return r0, c0
    return None


def _expected_output(inp: np.ndarray, frames: list[dict], anchors: list[dict]) -> np.ndarray:
    out = inp.copy()
    transformed = {
        int(frame["color"]): _rotate_cw_tight(frame["stamp"], int(frame["rotation"]))
        for frame in frames
    }
    for anchor in anchors:
        patch = transformed[int(anchor["color"])]
        _paste_non_bg(out, patch, int(anchor["row"]), int(anchor["col"]), _BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = int(_RNG.choice([4, 5]))
    frame_colors = [int(c) for c in _RNG.choice(np.arange(1, 10), size=frame_count, replace=False)]

    frames: list[dict] = []
    for i in range(frame_count):
        stamp = _make_stamp(i % 4)
        extra_side = int(_RNG.integers(4)) if _RNG.random() < 0.4 else -1
        pads = [1, 1, 1, 1]
        if extra_side >= 0:
            pads[extra_side] += 1
        top_pad, bottom_pad, left_pad, right_pad = pads
        frame_h = int(stamp.shape[0] + top_pad + bottom_pad + 2)
        frame_w = int(stamp.shape[1] + left_pad + right_pad + 2)
        if frame_h > 8 or frame_w > 8:
            return None
        anchor_count = int(_RNG.choice([1, 2]))
        frames.append(
            {
                "color": frame_colors[i],
                "stamp": stamp,
                "rotation": int(i % 4),
                "top_pad": top_pad,
                "bottom_pad": bottom_pad,
                "left_pad": left_pad,
                "right_pad": right_pad,
                "frame_h": frame_h,
                "frame_w": frame_w,
                "anchor_count": anchor_count,
            }
        )

    widths = [frame["frame_w"] for frame in frames]
    top_lane_max = max(frames[i]["frame_h"] for i in range(0, frame_count, 2))
    bottom_lane_max = max(frames[i]["frame_h"] for i in range(1, frame_count, 2))
    bottom_start = 2 + top_lane_max + 2 + bottom_lane_max + 2
    min_h = bottom_start + 6
    if min_h > 30:
        return None

    height = int(_RNG.integers(min_h, 31))
    width = int(_RNG.integers(25, 31))
    centers = _sample_centers(width, frame_count)
    if centers is None:
        return None

    top_lane_r0 = int(_RNG.integers(1, 3))
    bottom_lane_r0 = top_lane_r0 + top_lane_max + int(_RNG.integers(2, 4))
    if bottom_lane_r0 + bottom_lane_max + 7 >= height:
        return None

    grid = np.zeros((height, width), dtype=int)
    occ = np.zeros((height, width), dtype=bool)
    input_frames: list[dict] = []

    for i, frame in enumerate(frames):
        lane_r0 = top_lane_r0 if i % 2 == 0 else bottom_lane_r0
        frame_h = int(frame["frame_h"])
        frame_w = int(frame["frame_w"])
        c0 = int(centers[i] - frame_w // 2)
        r0 = int(lane_r0)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        if c0 < 1 or c1 >= width - 1:
            return None
        if not _rect_free(occ, r0, c0, frame_h, frame_w, pad=0):
            return None

        color = int(frame["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        stamp_top = r0 + 1 + int(frame["top_pad"])
        stamp_left = c0 + 1 + int(frame["left_pad"])
        _paste_non_bg(grid, frame["stamp"], stamp_top, stamp_left, _BG)
        _occupy_rect(occ, r0, c0, frame_h, frame_w, pad=1)

        input_frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "stamp": frame["stamp"],
                "rotation": frame["rotation"],
            }
        )

    anchors: list[dict] = []
    for frame in frames:
        patch = _rotate_cw_tight(frame["stamp"], int(frame["rotation"]))
        ph, pw = patch.shape
        for _ in range(int(frame["anchor_count"])):
            pos = _place_anchor_rect(occ, ph, pw)
            if pos is None:
                return None
            ar, ac = pos
            grid[ar, ac] = int(frame["color"])
            anchors.append({"color": int(frame["color"]), "row": ar, "col": ac})

    expected = _expected_output(grid, input_frames, anchors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
