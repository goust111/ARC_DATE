from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_SIDE = 30
GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(
    coords: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _components(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in (
                    (cr - 1, cc),
                    (cr + 1, cc),
                    (cr, cc - 1),
                    (cr, cc + 1),
                ):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": _mask_from_coords(coords, bbox),
                }
            )
    return comps


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    reps = []
    for rot in _rotations(mask):
        reps.append((rot.shape[0], rot.shape[1], *rot.astype(int).ravel().tolist()))
    return tuple(sorted(reps)[0:1])


def _same_shape_under_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    for rot in _rotations(a):
        if rot.shape == b.shape and np.array_equal(rot, b):
            return True
    return False


def _is_rect_frame(comp: dict[str, object]) -> bool:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 4 or w < 4:
        return False
    if int(mask.sum()) != 2 * h + 2 * w - 4:
        return False
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    return not np.any(mask[1:-1, 1:-1])


def _inside_frame_interior(
    comp: dict[str, object], frame_bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = frame_bbox
    coords = comp["coords"]
    return bool(
        np.all(coords[:, 0] > r0)
        and np.all(coords[:, 0] < r1)
        and np.all(coords[:, 1] > c0)
        and np.all(coords[:, 1] < c1)
    )


def _paste_mask(
    canvas: np.ndarray, mask: np.ndarray, color: int, top: int, left: int, bg: int
) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    assert patch.shape == mask.shape
    assert np.all(patch[mask] == bg)
    patch[mask] = color


def _parse_scene(
    g: np.ndarray,
) -> tuple[int, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    comps = _components(g, bg)

    frames = [comp for comp in comps if _is_rect_frame(comp)]
    frames.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    assert 2 <= len(frames) <= 3

    frame_ids = {id(comp) for comp in frames}
    objects = [comp for comp in comps if id(comp) not in frame_ids]

    samples_per_frame: list[list[dict[str, object]]] = [[] for _ in frames]
    fragments: list[dict[str, object]] = []
    for comp in objects:
        owners = [
            i
            for i, frame in enumerate(frames)
            if _inside_frame_interior(comp, frame["bbox"])
        ]
        if owners:
            assert len(owners) == 1
            samples_per_frame[owners[0]].append(comp)
        else:
            fragments.append(comp)

    assert fragments
    assert all(len(samples) == 1 for samples in samples_per_frame)
    samples = [samples[0] for samples in samples_per_frame]

    keys = [_shape_key(sample["mask"]) for sample in samples]
    assert len(set(keys)) == len(keys)
    return bg, frames, samples, fragments


def main(grid: np.ndarray) -> np.ndarray:
    """Classify rotated fragments by shape and pack them into the matching frame.

    Input structure
    - Background is black.
    - Two or three large grey rectangular frames appear on the grid.
    - Each frame already contains one small colored sample near its bottom.
    - Outside the frames are scattered monochrome fragments that match one sample
      shape up to rotation.

    Rule
    - Ignore color and match each scattered fragment to the frame whose sample has
      the same shape under 0/90/180/270-degree rotation.
    - Rotate the fragment to the sample's orientation.
    - Clear the fragment from its original position.
    - Inside the matching frame, place the rotated fragments immediately to the
      right of the sample, ordered by the fragments' original column positions.

    Key insight
    - Color is only an identity label. Shape under rotation determines ownership,
      and source position determines the packing order inside each frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, frames, samples, fragments = _parse_scene(g)
    out = g.copy()

    groups: list[list[dict[str, object]]] = [[] for _ in frames]
    for frag in fragments:
        matches = [
            i
            for i, sample in enumerate(samples)
            if _same_shape_under_rotation(frag["mask"], sample["mask"])
        ]
        assert len(matches) == 1
        groups[matches[0]].append(frag)
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for frame_idx, frags in enumerate(groups):
        frags.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
        sample = samples[frame_idx]
        sample_mask = sample["mask"]
        sh, sw = sample_mask.shape
        sr0, _sc0, _sr1, sc1 = sample["bbox"]
        _fr0, _fc0, _fr1, fc1 = frames[frame_idx]["bbox"]
        cursor = sc1 + 1
        for frag in frags:
            assert cursor + sw - 1 < fc1
            _paste_mask(out, sample_mask, int(frag["color"]), sr0, cursor, bg)
            cursor += sw

    return out


def _raw_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


RAW_SHAPES = (
    ("10", "11"),
    ("111", "010"),
    ("100", "111"),
    ("110", "011"),
    ("101", "111"),
    ("110", "010", "011"),
    ("111", "110"),
    ("011", "110", "010"),
)

SHAPE_LIBRARY: list[np.ndarray] = []
_SEEN_KEYS: set[tuple[tuple[int, int], ...]] = set()
for _rows in RAW_SHAPES:
    _mask = _raw_mask(_rows)
    _key = _shape_key(_mask)
    if _key not in _SEEN_KEYS:
        SHAPE_LIBRARY.append(_mask)
        _SEEN_KEYS.add(_key)


def _build_frame_patch(
    sample_mask: np.ndarray, sample_color: int, frag_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    sh, sw = sample_mask.shape
    extra_left = int(rng.integers(0, 2))
    extra_right = int(rng.integers(1, 3))
    extra_top = int(rng.integers(0, 3))
    interior_h = sh + extra_top
    interior_w = extra_left + sw * (1 + frag_count) + extra_right

    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    sample_top = 1 + interior_h - sh
    sample_left = 1 + extra_left
    _paste_mask(patch, sample_mask, sample_color, sample_top, sample_left, BG)
    sample_bbox = (sample_top, sample_left, sample_top + sh - 1, sample_left + sw - 1)
    return patch, sample_bbox


def _fragment_counts(n_boxes: int, rng: np.random.Generator) -> list[int]:
    if n_boxes == 2:
        total = int(rng.integers(2, 6))
        per_box_cap = 3
    else:
        total = int(rng.integers(3, 6))
        per_box_cap = 2
    counts = [1] * n_boxes
    extras = total - n_boxes
    while extras > 0:
        idx = int(rng.integers(0, n_boxes))
        if counts[idx] < per_box_cap:
            counts[idx] += 1
            extras -= 1
    return counts


def _layout_patterns(n_boxes: int) -> list[list[list[int]]]:
    if n_boxes == 2:
        return [[[0, 1]], [[0], [1]]]
    return [[[0, 1, 2]], [[0, 1], [2]], [[0], [1, 2]]]


def _choose_layout(
    frame_patches: list[np.ndarray],
    max_frag_h: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[tuple[int, int]], int, int, int] | None:
    n_boxes = len(frame_patches)
    widths = [patch.shape[1] for patch in frame_patches]
    heights = [patch.shape[0] for patch in frame_patches]
    perm = rng.permutation(n_boxes).tolist()

    candidate_rows: list[list[list[int]]] = []
    for pattern in _layout_patterns(n_boxes):
        rows = [[perm[i] for i in row] for row in pattern]
        candidate_rows.append(rows)

    valid_rows: list[list[list[int]]] = []
    for rows in candidate_rows:
        min_gap = 2
        row_widths = [
            sum(widths[idx] for idx in row) + min_gap * (len(row) - 1) for row in rows
        ]
        row_heights = [max(heights[idx] for idx in row) for row in rows]
        scatter_zone_h = max_frag_h + 5
        min_w = max(row_widths) + 2
        min_h = sum(row_heights) + 2 * (len(rows) - 1) + scatter_zone_h + 2
        if min_w <= MAX_SIDE and min_h <= MAX_SIDE:
            valid_rows.append(rows)
    if not valid_rows:
        return None

    rows = valid_rows[int(rng.integers(0, len(valid_rows)))]
    for _ in range(80):
        left = int(rng.integers(1, 4))
        right = int(rng.integers(1, 4))
        top = int(rng.integers(1, 4))
        bottom = int(rng.integers(1, 4))
        row_gap = int(rng.integers(2, 5))
        scatter_zone_h = max_frag_h + int(rng.integers(5, 9))

        row_widths = []
        row_heights = []
        row_gaps: list[list[int]] = []
        for row in rows:
            gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(row) - 1))]
            row_gaps.append(gaps)
            row_widths.append(
                sum(widths[idx] for idx in row) + sum(gaps)
            )
            row_heights.append(max(heights[idx] for idx in row))

        width = left + max(row_widths) + right
        height = top + sum(row_heights) + row_gap * (len(rows) - 1) + scatter_zone_h + bottom
        if width > MAX_SIDE or height > MAX_SIDE:
            continue

        positions: list[tuple[int, int]] = [(-1, -1)] * n_boxes
        y = top
        for row_idx, row in enumerate(rows):
            row_w = row_widths[row_idx]
            row_h = row_heights[row_idx]
            slack = width - left - right - row_w
            row_x = left + int(rng.integers(0, slack + 1))
            x = row_x
            for item_idx, frame_idx in enumerate(row):
                fh, fw = frame_patches[frame_idx].shape
                y_slack = row_h - fh
                frame_y = y + int(rng.integers(0, y_slack + 1))
                positions[frame_idx] = (frame_y, x)
                x += fw
                if item_idx < len(row) - 1:
                    x += row_gaps[row_idx][item_idx]
            y += row_h
            if row_idx < len(rows) - 1:
                y += row_gap
        scatter_top = y
        return rows, positions, width, height, scatter_top
    return None


def _mark_blocked(
    blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int
) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _find_fragment_spot(
    blocked: np.ndarray,
    scatter_top: int,
    mask: np.ndarray,
    banned_lefts: set[int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = mask.shape
    spots = []
    for top in range(scatter_top, h - ph + 1):
        for left in range(0, w - pw + 1):
            if left in banned_lefts:
                continue
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(h, top + ph + 1)
            c1 = min(w, left + pw + 1)
            if np.any(blocked[r0:r1, c0:c1]):
                continue
            spots.append((top, left))
    if not spots:
        return None
    return spots[int(rng.integers(0, len(spots)))]


def _build_expected(
    inp: np.ndarray,
    sample_infos: list[dict[str, object]],
    fragment_infos: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for frag in fragment_infos:
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    grouped: list[list[dict[str, object]]] = [[] for _ in sample_infos]
    for frag in fragment_infos:
        grouped[int(frag["box_idx"])].append(frag)

    for box_idx, frags in enumerate(grouped):
        frags.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        sample = sample_infos[box_idx]
        sample_mask = sample["mask"]
        sh, sw = sample_mask.shape
        sr0, _sc0, _sr1, sc1 = sample["bbox"]
        _frame_r0, _frame_c0, _frame_r1, frame_c1 = sample["frame_bbox"]
        cursor = sc1 + 1
        for frag in frags:
            assert cursor + sw - 1 < frame_c1
            _paste_mask(out, sample_mask, int(frag["color"]), sr0, cursor, BG)
            cursor += sw
    return out


def _generate_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    n_boxes = int(rng.integers(2, 4))
    counts = _fragment_counts(n_boxes, rng)

    shape_indices = rng.choice(len(SHAPE_LIBRARY), size=n_boxes, replace=False)
    sample_colors_and_frag_colors = rng.choice(
        np.array(COLOR_POOL, dtype=int), size=n_boxes + sum(counts), replace=False
    )
    sample_colors = sample_colors_and_frag_colors[:n_boxes].tolist()
    frag_colors = sample_colors_and_frag_colors[n_boxes:].tolist()

    sample_masks: list[np.ndarray] = []
    frame_patches: list[np.ndarray] = []
    sample_local_bboxes: list[tuple[int, int, int, int]] = []
    base_masks: list[np.ndarray] = []
    sample_rots: list[int] = []
    for box_idx in range(n_boxes):
        base = SHAPE_LIBRARY[int(shape_indices[box_idx])]
        sample_rot = int(rng.integers(0, 4))
        sample_mask = np.rot90(base, sample_rot)
        patch, local_bbox = _build_frame_patch(
            sample_mask, int(sample_colors[box_idx]), counts[box_idx], rng
        )
        base_masks.append(base)
        sample_rots.append(sample_rot)
        sample_masks.append(sample_mask)
        frame_patches.append(patch)
        sample_local_bboxes.append(local_bbox)

    fragment_specs = []
    frag_color_idx = 0
    for box_idx in range(n_boxes):
        for _ in range(counts[box_idx]):
            frag_rot = int(rng.integers(0, 4))
            fragment_specs.append(
                {
                    "box_idx": box_idx,
                    "color": int(frag_colors[frag_color_idx]),
                    "rot": frag_rot,
                }
            )
            frag_color_idx += 1

    if all(
        int(spec["rot"]) == sample_rots[int(spec["box_idx"])] for spec in fragment_specs
    ):
        chosen = int(rng.integers(0, len(fragment_specs)))
        fragment_specs[chosen]["rot"] = (int(fragment_specs[chosen]["rot"]) + 1) % 4

    max_frag_h = max(
        np.rot90(base_masks[int(spec["box_idx"])], int(spec["rot"])).shape[0]
        for spec in fragment_specs
    )
    layout = _choose_layout(frame_patches, max_frag_h, rng)
    if layout is None:
        return None
    _rows, frame_positions, width, height, scatter_top = layout

    inp = np.full((height, width), BG, dtype=int)
    sample_infos: list[dict[str, object]] = []
    blocked = np.zeros((height, width), dtype=bool)
    for box_idx, patch in enumerate(frame_patches):
        top, left = frame_positions[box_idx]
        ph, pw = patch.shape
        mask = patch != BG
        sub = inp[top : top + ph, left : left + pw]
        assert sub.shape == patch.shape
        assert np.all(sub[mask] == BG)
        sub[mask] = patch[mask]
        _mark_blocked(blocked, top, left, ph, pw, pad=1)

        sr0, sc0, sr1, sc1 = sample_local_bboxes[box_idx]
        sample_infos.append(
            {
                "mask": sample_masks[box_idx],
                "bbox": (top + sr0, left + sc0, top + sr1, left + sc1),
                "frame_bbox": (top, left, top + ph - 1, left + pw - 1),
            }
        )

    fragment_infos: list[dict[str, object]] = []
    used_lefts: list[set[int]] = [set() for _ in range(n_boxes)]
    order = rng.permutation(len(fragment_specs)).tolist()
    for spec_idx in order:
        spec = fragment_specs[spec_idx]
        box_idx = int(spec["box_idx"])
        frag_mask = np.rot90(base_masks[box_idx], int(spec["rot"]))
        spot = _find_fragment_spot(
            blocked, scatter_top, frag_mask, used_lefts[box_idx], rng
        )
        if spot is None:
            return None
        top, left = spot
        _paste_mask(inp, frag_mask, int(spec["color"]), top, left, BG)
        ph, pw = frag_mask.shape
        _mark_blocked(blocked, top, left, ph, pw, pad=1)
        used_lefts[box_idx].add(left)
        coords = np.argwhere(frag_mask) + np.array([top, left], dtype=int)
        fragment_infos.append(
            {
                "box_idx": box_idx,
                "color": int(spec["color"]),
                "coords": coords,
                "bbox": (top, left, top + ph - 1, left + pw - 1),
            }
        )

    expected = _build_expected(inp, sample_infos, fragment_infos)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _generate_case(rng)
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
