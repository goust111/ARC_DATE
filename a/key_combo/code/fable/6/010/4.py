from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATION_TO_SIDE = {0: "U", 1: "R", 2: "D", 3: "L"}
SIDE_TO_ORIENTATION = {side: orient for orient, side in ORIENTATION_TO_SIDE.items()}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = _coords(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(mask: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts = []
        while stack:
            pr, pc = stack.pop()
            pts.append((pr, pc))
            for dr, dc in FOUR_NEIGHBORS:
                nr = pr + dr
                nc = pc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    pts = int(mask.sum())
    if pts == 0:
        return False
    return len(_components(mask)) == 1


def _rotate_canonical_to_current(mask: np.ndarray, orientation: int) -> np.ndarray:
    return np.rot90(mask, k=(4 - int(orientation)) % 4)


def _rotate_current_to_canonical(mask: np.ndarray, orientation: int) -> np.ndarray:
    return np.rot90(mask, k=int(orientation))


def _subarray_offset_in_rotated_template(
    partial_shape: tuple[int, int],
    template_shape: tuple[int, int],
    orientation: int,
) -> tuple[int, int]:
    ph, pw = map(int, partial_shape)
    th, tw = map(int, template_shape)
    orientation = int(orientation)
    if orientation == 0:
        return 0, 0
    if orientation == 1:
        return 0, th - ph
    if orientation == 2:
        return th - ph, tw - pw
    if orientation == 3:
        return tw - pw, 0
    raise ValueError("invalid orientation")


def _marker_side_for_bbox(
    marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str | None:
    mr, mc = map(int, marker)
    r0, c0, r1, c1 = map(int, bbox)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h % 2 == 0 or w % 2 == 0:
        return None
    centers = {
        "U": (r0 - 1, c0 + w // 2),
        "R": (r0 + h // 2, c1 + 1),
        "D": (r1 + 1, c0 + w // 2),
        "L": (r0 + h // 2, c0 - 1),
    }
    for side, pos in centers.items():
        if pos == (mr, mc):
            return side
    return None


def _parse_fragments(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    markers = []
    for comp in _components(g == GREY):
        assert len(comp) == 1, "grey markers must be isolated singleton pixels"
        markers.append(tuple(map(int, comp[0])))
    assert len(markers) >= 2, "expected multiple marked fragments"
    marker_set = set(markers)

    pieces: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, GREY)):
        comps = _components(g == color)
        for comp in comps:
            mask = np.zeros(g.shape, dtype=bool)
            mask[comp[:, 0], comp[:, 1]] = True
            bbox = _bbox_from_mask(mask)
            assert bbox is not None
            side_matches = [
                side
                for marker in markers
                if (side := _marker_side_for_bbox(marker, bbox)) is not None
            ]
            assert len(side_matches) == 1, "each fragment bbox must have exactly one side marker"
            side = side_matches[0]
            if side == "U":
                marker = (bbox[0] - 1, bbox[1] + (bbox[3] - bbox[1] + 1) // 2)
            elif side == "R":
                marker = ((bbox[0] + bbox[2]) // 2, bbox[3] + 1)
            elif side == "D":
                marker = (bbox[2] + 1, bbox[1] + (bbox[3] - bbox[1] + 1) // 2)
            else:
                marker = ((bbox[0] + bbox[2]) // 2, bbox[1] - 1)
            assert marker in marker_set
            local = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == color
            orientation = SIDE_TO_ORIENTATION[side]
            canonical = _rotate_current_to_canonical(local, orientation)
            assert canonical[0, 0], "rotated fragment must retain the canonical top-left cell"
            pieces.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "marker": marker,
                    "orientation": orientation,
                    "current": local,
                    "canonical": canonical,
                }
            )

    assert len(pieces) == len(markers), "marker and fragment counts must match"
    used_markers = {piece["marker"] for piece in pieces}
    assert used_markers == marker_set, "every grey marker must belong to one fragment"
    return bg, pieces


def _recover_template(pieces: list[dict]) -> np.ndarray:
    template_h = max(int(piece["canonical"].shape[0]) for piece in pieces)
    template_w = max(int(piece["canonical"].shape[1]) for piece in pieces)
    template = np.zeros((template_h, template_w), dtype=bool)
    for piece in pieces:
        canonical = piece["canonical"]
        ph, pw = canonical.shape
        template[:ph, :pw] |= canonical
    assert template[0, 0], "the recovered template must keep the canonical top-left anchor"
    return template


def _embed_top_left(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[:h, :w] = mask
    return out


def _render_completed_output(grid: np.ndarray, bg: int, pieces: list[dict], template: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = bg
    th, tw = map(int, template.shape)
    for piece in pieces:
        orientation = int(piece["orientation"])
        canonical = piece["canonical"]
        ph, pw = canonical.shape
        completed = _rotate_canonical_to_current(template, orientation)
        off_r, off_c = _subarray_offset_in_rotated_template((ph, pw), (th, tw), orientation)
        r0, c0, _r1, _c1 = map(int, piece["bbox"])
        full_r0 = r0 - off_r
        full_c0 = c0 - off_c
        full_r1 = full_r0 + completed.shape[0]
        full_c1 = full_c0 + completed.shape[1]
        assert 0 <= full_r0 and 0 <= full_c0
        assert full_r1 <= out.shape[0] and full_c1 <= out.shape[1]
        region = out[full_r0:full_r1, full_c0:full_c1]
        assert np.all((region == bg) | (region == int(piece["color"]))), "completed fragments must not overlap"
        region[completed] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marked partial copy back to a canonical orientation, union them into the full template, then restore and complete every copy.

    Input structure:
    - A background grid contains several separated monochrome fragment copies of one common template.
    - Each fragment is a rotated and incomplete version of that template, using its own non-grey color.
    - A single grey pixel sits just outside the center of one side of each fragment's visible bounding box and marks which side is the fragment's top edge.

    Rule:
    - For every fragment, use the grey marker side to determine its orientation and rotate the fragment back to the canonical orientation.
    - Align those canonical fragment masks by their bounding-box top-left corners and take their union to recover the full template mask.
    - Rotate the recovered template back to each fragment's original orientation and place it so the original visible fragment stays fixed, filling all missing cells in that fragment's own color.
    - Remove all grey markers from the output.

    Key insight:
    - The fragments only complement each other after marker-guided rotational registration; without undoing the rotations first, their visible masks do not align to one shared template.
    """
    bg, pieces = _parse_fragments(np.asarray(grid, dtype=int))
    template = _recover_template(pieces)
    return _render_completed_output(np.asarray(grid, dtype=int), bg, pieces, template)


TEMPLATE_SHAPES = [(5, 7), (7, 5), (7, 7)]
GRID_SHAPE_CHOICES = [
    (17, 19),
    (18, 22),
    (19, 25),
    (20, 20),
    (21, 24),
    (22, 28),
    (24, 24),
    (25, 29),
]
COPY_COUNT_CHOICES = [3]
GEN_RETRIES = 400


def _random_connected_template(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = map(int, shape)
    right_witness = {(h // 2, w - 2), (h // 2, w - 1)}
    bottom_witness = {(h - 2, w // 2), (h - 1, w // 2)}
    corner_witness = {
        (h - 2, w - 3),
        (h - 2, w - 2),
        (h - 2, w - 1),
        (h - 1, w - 2),
        (h - 1, w - 1),
    }
    bottom_strip = {(h - 2, c) for c in range(w)} | {(h - 1, c) for c in range(w)}
    right_strip = {(r, w - 2) for r in range(h)} | {(r, w - 1) for r in range(h)}
    protected = {
        (0, 0),
        (0, w // 2),
        (h // 2, 0),
    } | bottom_witness | right_witness | corner_witness
    forbidden = {
        cell for cell in bottom_strip if cell not in bottom_witness | corner_witness
    } | {
        cell for cell in right_strip if cell not in right_witness | corner_witness
    }
    target = random.randint(max(9, (h * w) // 3), max(12, (2 * h * w) // 3))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, : w // 2 + 1] = True
    mask[: h // 2 + 1, 0] = True
    mask[:, w // 2] = True
    mask[h // 2, : w - 1] = True
    mask[h - 3, w // 2 : w - 2] = True
    for r, c in corner_witness:
        mask[r, c] = True
    for r, c in right_witness:
        mask[r, c] = True
    for r, c in bottom_witness:
        mask[r, c] = True
    for r, c in forbidden:
        mask[r, c] = False

    frontier = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in FOUR_NEIGHBORS:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in forbidden:
                frontier.add((nr, nc))

    while int(mask.sum()) < target and frontier:
        nr, nc = random.choice(sorted(frontier))
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        for dr, dc in FOUR_NEIGHBORS:
            rr = nr + dr
            cc = nc + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and (rr, cc) not in forbidden:
                frontier.add((rr, cc))

    if not protected.issubset({tuple(map(int, rc)) for rc in np.argwhere(mask)}):
        return None
    if not _is_connected(mask):
        return None
    if not mask[0, 0]:
        return None
    if {
        (r, c) for (r, c) in bottom_strip if mask[r, c]
    } != (bottom_witness | corner_witness):
        return None
    if {
        (r, c) for (r, c) in right_strip if mask[r, c]
    } != (right_witness | (corner_witness & right_strip)):
        return None
    for removed in (
        bottom_witness | corner_witness,
        right_witness | corner_witness,
        right_witness | bottom_witness,
    ):
        trial = mask.copy()
        for r, c in removed:
            trial[r, c] = False
        if not _is_connected(trial):
            return None
    if int(mask.sum()) < 11:
        return None
    if h == w:
        if np.array_equal(mask, np.rot90(mask, 1)):
            return None
        if np.array_equal(mask, np.rot90(mask, 2)):
            return None
    return mask


def _make_partial_without(template: np.ndarray, removed: set[tuple[int, int]]) -> np.ndarray | None:
    partial = template.copy()
    for r, c in removed:
        partial[r, c] = False
    if not partial[0, 0] or np.array_equal(partial, template):
        return None
    if not _is_connected(partial):
        return None
    return _crop_bbox(partial)


def _find_optional_extra_removed(
    template: np.ndarray,
    base_removed: set[tuple[int, int]],
    banned: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    random.shuffle(candidates)
    for cell in candidates:
        if cell in banned or cell in base_removed or cell == (0, 0):
            continue
        trial = _make_partial_without(template, base_removed | {cell})
        if trial is not None:
            return cell
    return None


def _build_partials(template: np.ndarray, count: int) -> list[np.ndarray] | None:
    h, w = template.shape
    if count != 3:
        return None
    right_witness = {(h // 2, w - 2), (h // 2, w - 1)}
    bottom_witness = {(h - 2, w // 2), (h - 1, w // 2)}
    corner_witness = {
        (h - 2, w - 3),
        (h - 2, w - 2),
        (h - 2, w - 1),
        (h - 1, w - 2),
        (h - 1, w - 1),
    }
    missing_sets = [
        set(bottom_witness | corner_witness),
        set(right_witness | corner_witness),
        set(right_witness | bottom_witness),
    ]
    banned = set(right_witness | bottom_witness | corner_witness)

    partials = []
    for missing in missing_sets:
        extra = None
        if random.random() < 0.75:
            extra = _find_optional_extra_removed(template, missing, banned)
        if extra is not None:
            missing = set(missing)
            missing.add(extra)
        partial = _make_partial_without(template, missing)
        if partial is None:
            return None
        partials.append(partial)

    union = np.zeros_like(template)
    for partial in partials:
        union |= _embed_top_left(partial, template.shape)
    if not np.array_equal(union, template):
        return None
    for i in range(len(partials)):
        for j in range(i + 1, len(partials)):
            pair_union = _embed_top_left(partials[i], template.shape) | _embed_top_left(partials[j], template.shape)
            if np.array_equal(pair_union, template):
                return None
    return partials


def _choose_orientations(count: int) -> list[int]:
    while True:
        choices = [random.randrange(4) for _ in range(count)]
        if len(set(choices)) >= 2 and any(v != 0 for v in choices):
            return choices


def _marker_position(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side == "U":
        return r0 - 1, c0 + w // 2
    if side == "R":
        return r0 + h // 2, c1 + 1
    if side == "D":
        return r1 + 1, c0 + w // 2
    return r0 + h // 2, c0 - 1


def _all_marker_positions_for_bbox(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return {_marker_position(bbox, side) for side in ("U", "R", "D", "L")}


def _box_clear(used: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(used.shape[0], r0 + h + margin)
    cc1 = min(used.shape[1], c0 + w + margin)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _reserve_box(used: np.ndarray, r0: int, c0: int, h: int, w: int, marker: tuple[int, int]) -> None:
    used[r0 : r0 + h, c0 : c0 + w] = True
    mr, mc = marker
    used[mr, mc] = True
    for dr, dc in FOUR_NEIGHBORS:
        nr = mr + dr
        nc = mc + dc
        if 0 <= nr < used.shape[0] and 0 <= nc < used.shape[1]:
            used[nr, nc] = True


def _naive_unaligned_union_shape(partials: list[np.ndarray], orientations: list[int]) -> np.ndarray:
    current_masks = [_rotate_canonical_to_current(partial, orient) for partial, orient in zip(partials, orientations)]
    h = max(mask.shape[0] for mask in current_masks)
    w = max(mask.shape[1] for mask in current_masks)
    union = np.zeros((h, w), dtype=bool)
    for mask in current_masks:
        ph, pw = mask.shape
        union[:ph, :pw] |= mask
    return union


def _assemble_instance() -> tuple[np.ndarray, np.ndarray]:
    template_shape = random.choice(TEMPLATE_SHAPES)
    template = _random_connected_template(template_shape)
    if template is None:
        raise ValueError("template generation failed")

    count = random.choice(COPY_COUNT_CHOICES)
    partials = _build_partials(template, count)
    if partials is None:
        raise ValueError("partial generation failed")

    orientations = _choose_orientations(count)
    if np.array_equal(_naive_unaligned_union_shape(partials, orientations), template):
        raise ValueError("unaligned union accidentally matches the canonical template")

    grid_h, grid_w = random.choice(GRID_SHAPE_CHOICES)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    colors = random.sample([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c != bg], count)

    items = []
    th, tw = template.shape
    for color, partial, orientation in zip(colors, partials, orientations):
        current = _rotate_canonical_to_current(partial, orientation)
        completed = _rotate_canonical_to_current(template, orientation)
        ph, pw = partial.shape
        off_r, off_c = _subarray_offset_in_rotated_template((ph, pw), (th, tw), orientation)
        items.append(
            {
                "color": color,
                "partial": partial,
                "orientation": orientation,
                "current": current,
                "completed": completed,
                "offset": (off_r, off_c),
            }
        )

    items.sort(key=lambda item: (item["completed"].shape[0] * item["completed"].shape[1]), reverse=True)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    oracle = np.full((grid_h, grid_w), bg, dtype=int)
    used = np.zeros((grid_h, grid_w), dtype=bool)
    placed_views: list[dict] = []

    for item in items:
        completed = item["completed"]
        current = item["current"]
        fh, fw = completed.shape
        ch, cw = current.shape
        off_r, off_c = item["offset"]
        placed = False
        candidates = [
            (r0, c0)
            for r0 in range(grid_h - fh + 1)
            for c0 in range(grid_w - fw + 1)
            if _box_clear(used, r0, c0, fh, fw, margin=1)
        ]
        random.shuffle(candidates)
        for full_r0, full_c0 in candidates:
            vis_r0 = full_r0 + off_r
            vis_c0 = full_c0 + off_c
            vis_bbox = (vis_r0, vis_c0, vis_r0 + ch - 1, vis_c0 + cw - 1)
            marker = _marker_position(vis_bbox, ORIENTATION_TO_SIDE[item["orientation"]])
            mr, mc = marker
            if not (0 <= mr < grid_h and 0 <= mc < grid_w):
                continue
            if used[mr, mc]:
                continue
            marker_positions = _all_marker_positions_for_bbox(vis_bbox)
            conflict = False
            for placed in placed_views:
                if marker in placed["marker_positions"]:
                    conflict = True
                    break
                if placed["marker"] in marker_positions:
                    conflict = True
                    break
            if conflict:
                continue
            current_mask = current.astype(bool)
            completed_mask = completed.astype(bool)
            if inp[mr, mc] != bg:
                continue
            if np.any(inp[vis_r0 : vis_r0 + ch, vis_c0 : vis_c0 + cw][current_mask] != bg):
                continue
            if np.any(oracle[full_r0 : full_r0 + fh, full_c0 : full_c0 + fw][completed_mask] != bg):
                continue
            inp[vis_r0 : vis_r0 + ch, vis_c0 : vis_c0 + cw][current_mask] = int(item["color"])
            inp[mr, mc] = GREY
            oracle[full_r0 : full_r0 + fh, full_c0 : full_c0 + fw][completed_mask] = int(item["color"])
            _reserve_box(used, full_r0, full_c0, fh, fw, marker)
            placed_views.append({"bbox": vis_bbox, "marker": marker, "marker_positions": marker_positions})
            item["placement"] = {
                "full_r0": full_r0,
                "full_c0": full_c0,
                "vis_r0": vis_r0,
                "vis_c0": vis_c0,
                "marker": marker,
            }
            placed = True
            break
        if not placed:
            raise ValueError("placement failed")

    _validate_instance(inp, oracle, template, partials, orientations)
    return inp, oracle


def _validate_instance(
    inp: np.ndarray,
    oracle: np.ndarray,
    template: np.ndarray,
    partials: list[np.ndarray],
    orientations: list[int],
) -> None:
    assert inp.ndim == 2 and oracle.ndim == 2
    assert max(inp.shape) <= 30 and max(oracle.shape) <= 30
    assert len(partials) >= 2
    assert len(set(orientations)) >= 2 and any(v != 0 for v in orientations)
    for partial in partials:
        assert partial[0, 0]
        assert _is_connected(partial)
        assert not np.array_equal(partial, template)
    union = np.zeros_like(template)
    for partial in partials:
        union |= _embed_top_left(partial, template.shape)
    assert np.array_equal(union, template)
    for i in range(len(partials)):
        for j in range(i + 1, len(partials)):
            pair_union = _embed_top_left(partials[i], template.shape) | _embed_top_left(partials[j], template.shape)
            assert not np.array_equal(pair_union, template)
    naive = _naive_unaligned_union_shape(partials, orientations)
    assert not np.array_equal(naive, template)
    solved = main(inp)
    assert np.array_equal(solved, oracle)

    bg = _get_bg(inp)
    grey_count = int(np.sum(inp == GREY))
    assert grey_count == len(partials)
    assert not np.any((oracle == GREY))
    piece_colors = sorted(int(v) for v in np.unique(inp) if int(v) not in (bg, GREY))
    assert len(piece_colors) == len(partials)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, _oracle = _assemble_instance()
            return inp
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
