from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 9
DIR_ORDER = ("up", "right", "down", "left")
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR_VECS.values():
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _rot_ccw(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return (-int(dc), int(dr))


def _rotated_side(side: str, steps: int) -> str:
    vec = DIR_VECS[side]
    for _ in range(int(steps) % 4):
        vec = _rot_ccw(vec)
    for name, cand in DIR_VECS.items():
        if cand == vec:
            return name
    raise AssertionError("invalid direction rotation")


def _ccw_steps(src_side: str, dst_side: str) -> int:
    cur = src_side
    for steps in range(4):
        if cur == dst_side:
            return int(steps)
        cur = _rotated_side(cur, 1)
    raise AssertionError("unreachable rotation")


def _parse_frame(coords: np.ndarray) -> dict | None:
    mask, bbox = _tight_mask(coords)
    h, w = mask.shape
    if h != w or h < 5:
        return None

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if bool(np.any(mask & ~border)):
        return None

    gap_sides: list[tuple[str, int]] = []
    side_masks = {
        "up": mask[0, :],
        "down": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }
    for side, side_mask in side_masks.items():
        missing = np.flatnonzero(~side_mask)
        if len(missing) == 0:
            continue
        if len(missing) not in (2, 3):
            return None
        if int(missing[-1]) - int(missing[0]) + 1 != len(missing):
            return None
        if int(missing[0]) == 0 or int(missing[-1]) == h - 1:
            return None
        gap_sides.append((side, int(len(missing))))

    if len(gap_sides) != 1:
        return None
    gap_side, gap_width = gap_sides[0]
    if int(mask.sum()) != int(border.sum()) - int(gap_width):
        return None
    return {
        "bbox": bbox,
        "mask": mask,
        "size": int(h),
        "gap_side": gap_side,
        "gap_width": int(gap_width),
    }


def _marker_neighbors(g: np.ndarray, coords: np.ndarray) -> set[tuple[int, int]]:
    h, w = g.shape
    found: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in DIR_VECS.values():
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == MARKER:
                found.add((rr, cc))
    return found


def _marker_side(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    mr, mc = map(int, marker)
    r0, c0, r1, c1 = bbox
    if mr == r0 - 1 and c0 <= mc <= c1:
        return "up"
    if mr == r1 + 1 and c0 <= mc <= c1:
        return "down"
    if mc == c0 - 1 and r0 <= mr <= r1:
        return "left"
    if mc == c1 + 1 and r0 <= mr <= r1:
        return "right"
    raise AssertionError("marker must sit just outside one side of the pattern bbox")


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = canvas[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    assert view.shape == mask.shape
    view[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored source pattern so its maroon-facing side matches the gap
    direction of the same-colored square frame, then place the pattern inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(c) for c in np.unique(g) if int(c) not in {BG, MARKER}]
    assert colors, "expected at least one frame-pattern color"

    out = np.full_like(g, BG)
    used_markers: set[tuple[int, int]] = set()

    for color in colors:
        comps = _components(g == int(color))
        assert len(comps) >= 2, "each color must have a frame and a source pattern"

        frame_idx = None
        frame_info = None
        for idx, comp in enumerate(comps):
            info = _parse_frame(comp)
            if info is None:
                continue
            assert frame_idx is None, "frame must be unique for each color"
            frame_idx = int(idx)
            frame_info = info

        assert frame_idx is not None and frame_info is not None, "failed to identify frame"
        pattern_comps = [comp for idx, comp in enumerate(comps) if idx != frame_idx]
        assert len(pattern_comps) == 1, "each color must have exactly one source pattern"
        pattern_coords = pattern_comps[0]
        pattern_mask, pattern_bbox = _tight_mask(pattern_coords)

        marker_candidates = _marker_neighbors(g, pattern_coords)
        assert len(marker_candidates) == 1, "pattern must have one adjacent maroon marker"
        marker = next(iter(marker_candidates))
        used_markers.add(marker)
        marker_side = _marker_side(marker, pattern_bbox)

        steps = _ccw_steps(marker_side, frame_info["gap_side"])
        rotated = np.rot90(pattern_mask, steps)

        fr0, fc0, fr1, fc1 = frame_info["bbox"]
        inner = int(frame_info["size"]) - 2
        rh, rw = map(int, rotated.shape)
        assert rh <= inner and rw <= inner, "rotated pattern must fit inside its frame"
        top = int(fr0 + 1 + (inner - rh) // 2)
        left = int(fc0 + 1 + (inner - rw) // 2)

        out[comps[frame_idx][:, 0], comps[frame_idx][:, 1]] = int(color)
        _paint_mask(out, rotated, top, left, color)

    all_markers = {tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)}
    assert used_markers == all_markers, "all maroon markers must belong to some pattern"
    return out


SHAPE_DIMS = ((3, 3), (4, 4), (2, 4), (4, 2))
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
OBJECT_COUNT_RANGE = (1, 4)
PACK_ROWS_MAX = 3
GEN_RETRIES = 400


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {(tuple(np.rot90(mask, k).shape), np.rot90(mask, k).tobytes()) for k in range(4)}


def _random_shape_mask() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(SHAPE_DIMS)
        area = random.randint(max(4, (h * w) // 2), h * w - 1)
        start = (random.randrange(h), random.randrange(w))
        chosen = {start}
        frontier = {start}
        while len(chosen) < area and frontier:
            base_r, base_c = random.choice(tuple(frontier))
            candidates = []
            for dr, dc in DIR_VECS.values():
                nr = int(base_r + dr)
                nc = int(base_c + dc)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                    candidates.append((nr, nc))
            if not candidates:
                frontier.remove((base_r, base_c))
                continue
            cell = random.choice(candidates)
            chosen.add(cell)
            frontier.add(cell)

        if len(chosen) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[int(r), int(c)] = True
        if not _touches_all_sides(mask):
            continue
        if len(_rotation_keys(mask)) < 4:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate source pattern")


def _make_sprite(mask: np.ndarray, color: int, marker_side: str) -> np.ndarray:
    h, w = mask.shape
    canvas = np.full((h + 2, w + 2), BG, dtype=int)
    inner = canvas[1 : 1 + h, 1 : 1 + w]
    inner[mask] = int(color)

    if marker_side == "up":
        options = np.flatnonzero(mask[0, :])
        canvas[0, 1 + int(random.choice(options.tolist()))] = MARKER
    elif marker_side == "down":
        options = np.flatnonzero(mask[-1, :])
        canvas[h + 1, 1 + int(random.choice(options.tolist()))] = MARKER
    elif marker_side == "left":
        options = np.flatnonzero(mask[:, 0])
        canvas[1 + int(random.choice(options.tolist())), 0] = MARKER
    else:
        options = np.flatnonzero(mask[:, -1])
        canvas[1 + int(random.choice(options.tolist())), w + 1] = MARKER

    coords = np.argwhere(canvas != BG)
    r0, c0, r1, c1 = _bbox(coords)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _make_frame(size: int, color: int, gap_side: str, gap_width: int) -> np.ndarray:
    frame = np.full((size, size), BG, dtype=int)
    frame[0, :] = int(color)
    frame[-1, :] = int(color)
    frame[:, 0] = int(color)
    frame[:, -1] = int(color)
    start = (size - gap_width) // 2
    if gap_side == "up":
        frame[0, start : start + gap_width] = BG
    elif gap_side == "down":
        frame[-1, start : start + gap_width] = BG
    elif gap_side == "left":
        frame[start : start + gap_width, 0] = BG
    else:
        frame[start : start + gap_width, -1] = BG
    return frame


def _crop_non_bg(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0, r1, c1 = _bbox(coords)
    return inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]


def _build_cluster(color: int) -> tuple[np.ndarray, np.ndarray]:
    pattern = _random_shape_mask()
    ph, pw = map(int, pattern.shape)
    marker_side = random.choice(DIR_ORDER)
    steps = random.randrange(4)
    gap_side = _rotated_side(marker_side, steps)
    gap_width = random.choice((2, 3))
    near_side = random.choice(DIR_ORDER)
    sprite = _make_sprite(pattern, color, marker_side)
    sh, sw = map(int, sprite.shape)

    inner = int(max(ph, pw) + 2)
    if ph % 2 == 1 and pw % 2 == 1 and random.random() < 0.5:
        inner += 2
    frame_size = inner + 2
    frame = _make_frame(frame_size, color, gap_side, gap_width)
    sep = random.randint(1, 3)

    if near_side in {"up", "down"}:
        canvas_h = frame_size + sh + sep
        canvas_w = max(frame_size, sw) + 2
        frame_r0 = sh + sep if near_side == "up" else 0
        sprite_r0 = 0 if near_side == "up" else frame_size + sep
        frame_c0 = (canvas_w - frame_size) // 2
        sprite_pref = frame_c0 + (frame_size - sw) // 2 + random.randint(-1, 1)
        sprite_c0 = min(max(sprite_pref, 0), canvas_w - sw)
    else:
        canvas_h = max(frame_size, sh) + 2
        canvas_w = frame_size + sw + sep
        frame_c0 = sw + sep if near_side == "left" else 0
        sprite_c0 = 0 if near_side == "left" else frame_size + sep
        frame_r0 = (canvas_h - frame_size) // 2
        sprite_pref = frame_r0 + (frame_size - sh) // 2 + random.randint(-1, 1)
        sprite_r0 = min(max(sprite_pref, 0), canvas_h - sh)

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    exp = np.full((canvas_h, canvas_w), BG, dtype=int)

    _paint_mask(inp, frame != BG, frame_r0, frame_c0, color)
    _paint_mask(exp, frame != BG, frame_r0, frame_c0, color)
    patch_view = inp[sprite_r0 : sprite_r0 + sh, sprite_c0 : sprite_c0 + sw]
    sprite_mask = sprite != BG
    patch_view[sprite_mask] = sprite[sprite_mask]

    rotated = np.rot90(pattern, steps)
    rh, rw = map(int, rotated.shape)
    top = int(frame_r0 + 1 + (inner - rh) // 2)
    left = int(frame_c0 + 1 + (inner - rw) // 2)
    assert (inner - rh) % 2 == 0 and (inner - rw) % 2 == 0
    _paint_mask(exp, rotated, top, left, color)
    return _crop_non_bg(inp, exp)


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _split_nonempty(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals = [cuts[0]]
    for i in range(1, len(cuts)):
        vals.append(cuts[i] - cuts[i - 1])
    vals.append(total - cuts[-1])
    return [int(v) for v in vals]


def _pack_clusters(clusters: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    order = list(range(len(clusters)))
    random.shuffle(order)
    n_rows = random.randint(1, min(PACK_ROWS_MAX, len(clusters)))
    counts = _split_nonempty(len(clusters), n_rows)

    rows: list[dict] = []
    idx = 0
    for count in counts:
        members = order[idx : idx + count]
        idx += count
        row_gaps = [random.randint(1, 3) for _ in range(max(0, count - 1))]
        widths = [int(clusters[m][0].shape[1]) for m in members]
        heights = [int(clusters[m][0].shape[0]) for m in members]
        row_w = sum(widths) + sum(row_gaps)
        row_h = max(heights)
        rows.append({"members": members, "gaps": row_gaps, "width": row_w, "height": row_h})

    row_gaps = [random.randint(1, 3) for _ in range(max(0, n_rows - 1))]
    margin_top = random.randint(0, 2)
    margin_bottom = random.randint(0, 2)
    margin_left = random.randint(0, 2)
    margin_right = random.randint(0, 2)
    canvas_h = margin_top + margin_bottom + sum(int(row["height"]) for row in rows) + sum(row_gaps)
    canvas_w = margin_left + margin_right + max(int(row["width"]) for row in rows)
    if canvas_h > 30 or canvas_w > 30:
        return None

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    exp = np.full((canvas_h, canvas_w), BG, dtype=int)
    cur_r = margin_top
    for row_idx, row in enumerate(rows):
        slack_x = canvas_w - margin_left - margin_right - int(row["width"])
        cur_c = margin_left + (random.randint(0, slack_x) if slack_x > 0 else 0)
        for item_idx, member in enumerate(row["members"]):
            patch_in, patch_out = clusters[member]
            ph, pw = map(int, patch_in.shape)
            top = cur_r + random.randint(0, int(row["height"]) - ph)
            _paste_patch(inp, patch_in, top, cur_c)
            _paste_patch(exp, patch_out, top, cur_c)
            cur_c += pw
            if item_idx < len(row["gaps"]):
                cur_c += int(row["gaps"][item_idx])
        if row_idx < len(row_gaps):
            cur_r += int(row["height"]) + int(row_gaps[row_idx])
    return inp, exp


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(FRAME_COLORS, n_objects)
    clusters = [_build_cluster(int(color)) for color in colors]
    packed = _pack_clusters(clusters)
    if packed is None:
        return None
    inp, exp = packed
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, exp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate valid input")
