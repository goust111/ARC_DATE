import random
from collections import Counter, deque

import numpy as np


GENERATE_MAX_RETRIES = 500
MAX_FRAME_SIZE = 6
MIN_FRAME_SIZE = 4
GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)
PADDING = 1


AXIS_SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=bool),
]

ROT180_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]], dtype=bool),
]

NONE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
]


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _trim_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    return _trim_to_mask(patch, patch != bg)


def _trim_to_mask(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=int)
    return patch[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(coords, dtype=int))
    return comps


def _nonbg_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    return _components_mask(grid != bg)


def _mono_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components_mask(grid == color)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    m = _trim_mask(mask)
    return m.shape, tuple(int(x) for x in m.ravel())


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    keys = []
    for k in range(4):
        keys.append(_mask_key(np.rot90(mask, k)))
    return min(keys)


def _piece_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    return patch != bg


def _symmetry_flags(patch: np.ndarray, bg: int) -> tuple[bool, bool, bool]:
    t = _trim_patch(patch, bg)
    lr = bool(np.array_equal(t, t[:, ::-1]))
    ud = bool(np.array_equal(t, t[::-1, :]))
    rot180 = bool(np.array_equal(t, t[::-1, ::-1]))
    return lr, ud, rot180


def _classify_patch(patch: np.ndarray, bg: int) -> str:
    lr, ud, rot180 = _symmetry_flags(patch, bg)
    if lr and not ud and not rot180:
        return "lr"
    if ud and not lr and not rot180:
        return "ud"
    if rot180 and not lr and not ud:
        return "rot180"
    return "none"


def _topmost_color(patch: np.ndarray, bg: int) -> int:
    mask = patch != bg
    coords = np.argwhere(mask)
    r = int(coords[:, 0].min())
    c = int(coords[coords[:, 0] == r][:, 1].min())
    return int(patch[r, c])


def _rotated_patch(patch: np.ndarray, bg: int, k: int) -> np.ndarray:
    return _trim_patch(np.rot90(patch, k), bg)


def _matching_rotations(patch: np.ndarray, gap_mask: np.ndarray, bg: int) -> list[int]:
    want = _trim_mask(gap_mask)
    ks = []
    for k in range(4):
        rp = _rotated_patch(patch, bg, k)
        if np.array_equal(rp != bg, want):
            ks.append(k)
    return ks


def _gap_side(mask_in_bbox: np.ndarray) -> str | None:
    touches = {
        "top": bool(mask_in_bbox[0].any()),
        "bottom": bool(mask_in_bbox[-1].any()),
        "left": bool(mask_in_bbox[:, 0].any()),
        "right": bool(mask_in_bbox[:, -1].any()),
    }
    sides = [name for name, flag in touches.items() if flag]
    if len(sides) != 1:
        return None
    return sides[0]


def _choose_rotation(patch: np.ndarray, gap_mask: np.ndarray, side: str, frame_color: int, bg: int) -> int:
    category = _classify_patch(patch, bg)
    candidates = _matching_rotations(patch, gap_mask, bg)
    assert candidates, "piece must fit the frame gap under some rotation"

    if category == "lr":
        need_lr = side in ("top", "bottom")
        filtered = []
        for k in candidates:
            rp = _rotated_patch(patch, bg, k)
            lr, ud, _ = _symmetry_flags(rp, bg)
            if (need_lr and lr) or ((not need_lr) and ud):
                filtered.append(k)
        candidates = filtered or candidates
    elif category == "ud":
        need_ud = side in ("top", "bottom")
        filtered = []
        for k in candidates:
            rp = _rotated_patch(patch, bg, k)
            lr, ud, _ = _symmetry_flags(rp, bg)
            if (need_ud and ud) or ((not need_ud) and lr):
                filtered.append(k)
        candidates = filtered or candidates
    elif category == "rot180":
        filtered = []
        for k in candidates:
            rp = _rotated_patch(patch, bg, k)
            if _topmost_color(rp, bg) == int(frame_color):
                filtered.append(k)
        candidates = filtered or candidates
    else:
        if 0 in candidates:
            return 0

    return min(candidates)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    h, w = patch.shape
    dst_slice = dst[r0 : r0 + h, c0 : c0 + w]
    dst_slice[mask] = patch[mask]


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frames = []
    frame_mask = np.zeros_like(grid, dtype=bool)
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _mono_components(grid, color):
            if coords.shape[0] < 8:
                continue
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < MIN_FRAME_SIZE or w < MIN_FRAME_SIZE:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if set(map(int, np.unique(sub))) - {bg, color}:
                continue
            present = sub == color
            if int(present.sum()) != int(coords.shape[0]):
                continue
            gap = ~present
            if not gap.any():
                continue
            gap_comps = _components_mask(gap)
            if len(gap_comps) != 1:
                continue
            side = _gap_side(gap)
            if side is None:
                continue
            gr0, gc0, gr1, gc1 = _bbox_from_coords(gap_comps[0])
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap_side": side,
                    "gap_offset": (gr0, gc0),
                    "gap_mask": gap[gr0 : gr1 + 1, gc0 : gc1 + 1].copy(),
                    "shape_key": _canonical_shape_key(gap),
                    "coords": coords,
                }
            )
            frame_mask[coords[:, 0], coords[:, 1]] = True
    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["color"]))
    return frames, frame_mask


def _find_pieces(grid: np.ndarray, bg: int, frame_cells: np.ndarray) -> list[dict]:
    residual = grid.copy()
    residual[frame_cells] = bg
    pieces = []
    for coords in _nonbg_components(residual, bg):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = residual[r0 : r1 + 1, c0 : c1 + 1].copy()
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords.copy(),
                "patch": patch,
                "shape_key": _canonical_shape_key(patch != bg),
            }
        )
    pieces.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据框架内部的缺口形状选择匹配的填充碎片，再按碎片自身颜色分布的对称性类型决定放置朝向。

    输入结构:
    - 背景色填充大部分网格。
    - 若干大型单色矩形框架；每个框架在一条边上有一个由背景色形成的缺口。
    - 若干小型彩色碎片散落在别处；每个碎片的外轮廓与某个缺口匹配。

    规则:
    - 先按外轮廓在旋转意义下匹配碎片与缺口。
    - 再按碎片当前颜色分布的对称性决定放入缺口时的旋转:
      左右对称 -> 对称轴与缺口所在边垂直。
      上下对称 -> 对称轴与缺口所在边平行。
      180 度旋转对称 -> 选最上方像素颜色与框架颜色相同的朝向。
      无对称 -> 保持原始朝向。
    - 填入后，碎片原位置清为空背景。
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames, frame_cells = _find_frames(g, bg)
    pieces = _find_pieces(g, bg, frame_cells)

    assert frames, "expected at least one frame"
    assert len(frames) == len(pieces), "piece count must match frame count"

    frame_by_key: dict[tuple[tuple[int, int], tuple[int, ...]], dict] = {}
    for frame in frames:
        key = _canonical_shape_key(frame["gap_mask"])
        assert key not in frame_by_key, "shape pairing must be unique"
        frame_by_key[key] = frame

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg
        frame = frame_by_key[piece["shape_key"]]
        k = _choose_rotation(piece["patch"], frame["gap_mask"], frame["gap_side"], frame["color"], bg)
        rp = _rotated_patch(piece["patch"], bg, k)
        r0, c0, _, _ = frame["bbox"]
        gr0, gc0 = frame["gap_offset"]
        assert np.array_equal(rp != bg, frame["gap_mask"])
        _paste(out, rp, int(r0 + gr0), int(c0 + gc0), bg)
    return out


def _orbit_partition(mask: np.ndarray, mode: str) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    used = set()
    groups = []
    for rc in sorted(coords):
        if rc in used:
            continue
        r, c = rc
        if mode == "lr":
            mate = (r, w - 1 - c)
        elif mode == "ud":
            mate = (h - 1 - r, c)
        else:
            mate = (h - 1 - r, w - 1 - c)
        orbit = {rc}
        if mate in coords:
            orbit.add(mate)
        used |= orbit
        groups.append(sorted(orbit))
    return groups


def _random_color_assignment(mask: np.ndarray, colors: list[int], groups: list[list[tuple[int, int]]]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    for group in groups:
        color = random.choice(colors)
        for r, c in group:
            patch[r, c] = color
    return patch


def _make_axis_patch(mask: np.ndarray, bg: int, want: str, colors: list[int]) -> np.ndarray | None:
    groups = _orbit_partition(mask, want)
    for _ in range(200):
        body = _random_color_assignment(mask, colors, groups)
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = body[mask]
        if len(set(map(int, patch[mask]))) < 2:
            continue
        if _classify_patch(patch, bg) == want:
            return patch
    return None


def _make_rot180_patch(mask: np.ndarray, bg: int, colors: list[int], frame_color: int, want_group: int) -> np.ndarray | None:
    groups = _orbit_partition(mask, "rot180")
    all_colors = list(dict.fromkeys([frame_color] + colors))
    for _ in range(400):
        body = _random_color_assignment(mask, all_colors, groups)
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = body[mask]
        if len(set(map(int, patch[mask]))) < 2:
            continue
        if _classify_patch(patch, bg) != "rot180":
            continue
        top0 = _topmost_color(patch, bg)
        top1 = _topmost_color(_rotated_patch(patch, bg, 1), bg)
        cond0 = top0 == frame_color
        cond1 = top1 == frame_color
        if cond0 == cond1:
            continue
        if (want_group == 0 and cond0) or (want_group == 1 and cond1):
            return patch
    return None


def _make_none_patch(mask: np.ndarray, bg: int, colors: list[int]) -> np.ndarray | None:
    coords = np.argwhere(mask)
    for _ in range(300):
        patch = np.full(mask.shape, bg, dtype=int)
        for r, c in coords:
            patch[int(r), int(c)] = random.choice(colors)
        if len(set(map(int, patch[mask]))) < 2:
            continue
        if _classify_patch(patch, bg) == "none":
            return patch
    return None


def _rule_rotation_for_generated_piece(category: str, side: str) -> int:
    if category == "lr":
        return 0 if side in ("top", "bottom") else 1
    if category == "ud":
        return 0 if side in ("top", "bottom") else 1
    if category == "rot180":
        return random.choice([0, 1])
    return 0


def _shape_pool_for_category(category: str) -> list[np.ndarray]:
    if category in ("lr", "ud"):
        return AXIS_SHAPES
    if category == "rot180":
        return ROT180_SHAPES
    return NONE_SHAPES


def _observed_patch_for_category(
    category: str,
    base_mask: np.ndarray,
    bg: int,
    frame_color: int,
    free_colors: list[int],
) -> np.ndarray | None:
    if category == "lr":
        obs_mask = np.rot90(base_mask, random.choice([0, 2]))
        return _make_axis_patch(obs_mask, bg, "lr", free_colors)
    if category == "ud":
        obs_mask = np.rot90(base_mask, random.choice([1, 3]))
        return _make_axis_patch(obs_mask, bg, "ud", free_colors)
    if category == "rot180":
        obs_mask = np.rot90(base_mask, random.choice([0, 1, 2, 3]))
        want_group = random.choice([0, 1])
        patch = _make_rot180_patch(obs_mask, bg, free_colors, frame_color, want_group)
        if patch is not None:
            patch.flags.writeable = True
        return patch
    obs_mask = np.rot90(base_mask, random.choice([0, 1, 2, 3]))
    return _make_none_patch(obs_mask, bg, free_colors)


def _frame_geometry(mask: np.ndarray, side: str) -> tuple[int, int, int, int] | None:
    gh, gw = mask.shape
    min_h = max(MIN_FRAME_SIZE, gh + (1 if side in ("top", "bottom") else 2))
    min_w = max(MIN_FRAME_SIZE, gw + (2 if side in ("top", "bottom") else 1))
    if min_h > MAX_FRAME_SIZE or min_w > MAX_FRAME_SIZE:
        return None
    fh = random.randint(min_h, MAX_FRAME_SIZE)
    fw = random.randint(min_w, MAX_FRAME_SIZE)
    if side == "top":
        gr0 = 0
        gc0 = random.randint(1, fw - gw - 1)
    elif side == "bottom":
        gr0 = fh - gh
        gc0 = random.randint(1, fw - gw - 1)
    elif side == "left":
        gc0 = 0
        gr0 = random.randint(1, fh - gh - 1)
    else:
        gc0 = fw - gw
        gr0 = random.randint(1, fh - gh - 1)
    return fh, fw, gr0, gc0


def _build_frame(frame_color: int, fh: int, fw: int, gap_mask: np.ndarray, gr0: int, gc0: int, bg: int) -> np.ndarray:
    frame = np.full((fh, fw), frame_color, dtype=int)
    sub = frame[gr0 : gr0 + gap_mask.shape[0], gc0 : gc0 + gap_mask.shape[1]]
    sub[gap_mask] = bg
    return frame


def _pick_position(reserved: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    H, W = reserved.shape
    options = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            rr0 = max(0, r - pad)
            rr1 = min(H, r + h + pad)
            cc0 = max(0, c - pad)
            cc1 = min(W, c + w + pad)
            if not reserved[rr0:rr1, cc0:cc1].any():
                options.append((r, c))
    if not options:
        return None
    return random.choice(options)


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    n = random.randint(2, 4)
    categories = random.sample(["lr", "ud", "rot180", "none"], n)

    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    frame_colors = palette[:n]
    piece_palette = palette[n:]
    if len(piece_palette) < 3:
        return None

    used_keys = set()
    specs = []
    for idx, category in enumerate(categories):
        candidates = []
        for base in _shape_pool_for_category(category):
            key = _canonical_shape_key(base)
            if key not in used_keys:
                candidates.append(base)
        if not candidates:
            return None
        base_mask = random.choice(candidates)
        used_keys.add(_canonical_shape_key(base_mask))

        frame_color = frame_colors[idx]
        free_colors = [c for c in piece_palette if c != frame_color]
        if len(free_colors) < 2:
            return None
        observed = _observed_patch_for_category(category, base_mask, bg, frame_color, free_colors)
        if observed is None:
            return None
        observed = _trim_patch(observed, bg)
        observed_category = _classify_patch(observed, bg)
        if observed_category != category:
            return None

        side = random.choice(["top", "bottom", "left", "right"])
        k_place = _rule_rotation_for_generated_piece(category, side)
        if category == "rot180":
            valid_groups = []
            for k in (0, 1):
                rotated = _rotated_patch(observed, bg, k)
                if _topmost_color(rotated, bg) == frame_color:
                    valid_groups.append(k)
            if len(valid_groups) != 1:
                return None
            k_place = valid_groups[0]
        gap_patch = _rotated_patch(observed, bg, k_place)
        gap_mask = gap_patch != bg

        geom = _frame_geometry(gap_mask, side)
        if geom is None:
            return None
        fh, fw, gr0, gc0 = geom
        frame_arr = _build_frame(frame_color, fh, fw, gap_mask, gr0, gc0, bg)
        if _gap_side(frame_arr == bg) != side:
            return None

        specs.append(
            {
                "category": category,
                "frame_color": frame_color,
                "side": side,
                "frame_arr": frame_arr,
                "gap_offset": (gr0, gc0),
                "piece_patch": observed,
                "placed_patch": gap_patch,
                "shape_key": _canonical_shape_key(gap_mask),
            }
        )

    reserved = np.zeros((H, W), dtype=bool)
    for spec in specs:
        fh, fw = spec["frame_arr"].shape
        pos = _pick_position(reserved, fh, fw, PADDING + 1)
        if pos is None:
            return None
        spec["frame_pos"] = pos
        _reserve_box(reserved, pos[0], pos[1], fh, fw, PADDING + 1)

    for spec in specs:
        ph, pw = spec["piece_patch"].shape
        pos = _pick_position(reserved, ph, pw, PADDING)
        if pos is None:
            return None
        spec["piece_pos"] = pos
        _reserve_box(reserved, pos[0], pos[1], ph, pw, PADDING)

    inp = np.full((H, W), bg, dtype=int)
    for spec in specs:
        fr, fc = spec["frame_pos"]
        fh, fw = spec["frame_arr"].shape
        inp[fr : fr + fh, fc : fc + fw] = spec["frame_arr"]

    for spec in specs:
        pr, pc = spec["piece_pos"]
        _paste(inp, spec["piece_patch"], pr, pc, bg)

    if _bg_color(inp) != bg:
        return None

    out = inp.copy()
    for spec in specs:
        pr, pc = spec["piece_pos"]
        piece = spec["piece_patch"]
        mask = piece != bg
        out[pr : pr + piece.shape[0], pc : pc + piece.shape[1]][mask] = bg

        fr, fc = spec["frame_pos"]
        gr0, gc0 = spec["gap_offset"]
        _paste(out, spec["placed_patch"], fr + gr0, fc + gc0, bg)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
