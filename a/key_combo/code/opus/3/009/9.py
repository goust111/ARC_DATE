from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
SPRITE_COLORS = [2, 3, 4, 5, 6, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GENERATE_ATTEMPTS = 400
SPRITE_ATTEMPTS = 500
TEMPLATE_ATTEMPTS = 300


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neigh = NEIGHBORS4
    else:
        neigh = NEIGHBORS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components_mask(mask, connectivity=4)) == 1


def _axis_score(mask: np.ndarray, axis: str) -> int:
    flipped = np.fliplr(mask) if axis == "vertical" else np.flipud(mask)
    relevant = mask | flipped
    return int(np.count_nonzero((mask == flipped) & relevant))


def _detect_defects(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    defects: list[tuple[int, int]] = []
    if axis == "vertical":
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if c >= mc:
                    break
                left = bool(mask[r, c])
                right = bool(mask[r, mc])
                if left and not right:
                    defects.append((r, mc))
                elif right and not left:
                    defects.append((r, c))
    else:
        for r in range(h):
            mr = h - 1 - r
            if r >= mr:
                break
            for c in range(w):
                top = bool(mask[r, c])
                bottom = bool(mask[mr, c])
                if top and not bottom:
                    defects.append((mr, c))
                elif bottom and not top:
                    defects.append((r, c))
    return sorted(defects)


def _best_axis(mask: np.ndarray) -> tuple[str, int, int]:
    v_score = _axis_score(mask, "vertical")
    h_score = _axis_score(mask, "horizontal")
    if v_score > h_score:
        return "vertical", v_score, h_score
    if h_score > v_score:
        return "horizontal", v_score, h_score
    v_defects = len(_detect_defects(mask, "vertical"))
    h_defects = len(_detect_defects(mask, "horizontal"))
    if v_defects >= h_defects:
        return "vertical", v_score, h_score
    return "horizontal", v_score, h_score


def _decorate_from_defects(
    raw_patch: np.ndarray, color: int, defects: Iterable[tuple[int, int]]
) -> np.ndarray:
    out = np.array(raw_patch, dtype=int, copy=True)
    h, w = out.shape
    for r, c in defects:
        for dr, dc in NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == BLACK:
                out[rr, cc] = ORANGE
    for r, c in defects:
        out[r, c] = int(color)
    return out


def _process_sprite_patch(raw_patch: np.ndarray, color: int, axis: str) -> np.ndarray:
    mask = raw_patch == int(color)
    defects = _detect_defects(mask, axis)
    return _decorate_from_defects(raw_patch, color, defects)


def _centered_interval_mask(widths: list[int], w: int) -> np.ndarray:
    h = len(widths)
    mask = np.zeros((h, w), dtype=bool)
    for r, width in enumerate(widths):
        start = (w - width) // 2
        mask[r, start : start + width] = True
    return mask


def _make_vertical_sprite(
    h: int, w: int, color: int, defect_count: int
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    allowed_widths = [width for width in range(1, w + 1) if width % 2 == w % 2]
    for _ in range(SPRITE_ATTEMPTS):
        widths = [random.choice(allowed_widths) for _ in range(h)]
        if w not in widths:
            continue
        if widths == widths[::-1]:
            continue
        candidate_rows = [r for r in range(1, h - 1) if widths[r] < w]
        if len(candidate_rows) < defect_count:
            continue
        final_area = sum(widths) - defect_count
        if not (10 <= final_area <= 20):
            continue

        raw_mask = _centered_interval_mask(widths, w)
        chosen_rows = random.sample(candidate_rows, defect_count)
        defects: list[tuple[int, int]] = []
        for r in chosen_rows:
            start = (w - widths[r]) // 2
            edge_cells = [start, start + widths[r] - 1]
            c = random.choice(edge_cells)
            raw_mask[r, c] = False
            defects.append((r, c))

        if not _is_contiguous(raw_mask):
            continue
        if _bbox_coords(np.argwhere(raw_mask)) != (0, 0, h - 1, w - 1):
            continue

        defects = sorted(defects)
        if _detect_defects(raw_mask, "vertical") != defects:
            continue
        v_score = _axis_score(raw_mask, "vertical")
        h_score = _axis_score(raw_mask, "horizontal")
        if v_score - h_score < 2:
            continue

        raw_patch = np.zeros((h, w), dtype=int)
        raw_patch[raw_mask] = int(color)
        processed = _decorate_from_defects(raw_patch, color, defects)
        if np.count_nonzero(processed == ORANGE) < defect_count:
            continue

        return raw_patch, processed, {
            "axis": "vertical",
            "defects": defects,
            "score_gap": int(v_score - h_score),
        }
    return None


def _make_horizontal_sprite(
    h: int, w: int, color: int, defect_count: int
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    for _ in range(SPRITE_ATTEMPTS):
        made = _make_vertical_sprite(w, h, color, defect_count)
        if made is None:
            continue
        raw_t, _processed_t, meta_t = made
        raw_patch = raw_t.T.copy()
        defects = sorted((c, r) for r, c in meta_t["defects"])
        processed = _decorate_from_defects(raw_patch, color, defects)
        mask = raw_patch == int(color)
        if _detect_defects(mask, "horizontal") != defects:
            continue
        v_score = _axis_score(mask, "vertical")
        h_score = _axis_score(mask, "horizontal")
        if h_score - v_score < 2:
            continue
        return raw_patch, processed, {
            "axis": "horizontal",
            "defects": defects,
            "score_gap": int(h_score - v_score),
        }
    return None


def _make_sprite(
    h: int, w: int, color: int
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    defect_count = random.choice([1, 2])
    axis = random.choice(["vertical", "horizontal"])
    if axis == "vertical":
        made = _make_vertical_sprite(h, w, color, defect_count)
    else:
        made = _make_horizontal_sprite(h, w, color, defect_count)
    if made is None:
        return None
    raw_patch, processed, meta = made
    meta = {
        "color": int(color),
        "shape": tuple(int(x) for x in raw_patch.shape),
        "raw": raw_patch,
        "processed": processed,
        **meta,
    }
    return raw_patch, processed, meta


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r - pad)
    rr1 = min(blocked.shape[0], r + h + pad)
    cc0 = max(0, c - pad)
    cc1 = min(blocked.shape[1], c + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _make_template_patch(
    hole_dims: list[tuple[int, int]]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]]] | None:
    max_h = max(h for h, _ in hole_dims)
    max_w = max(w for _, w in hole_dims)
    min_h = max(12, max_h + 4)
    min_w = max(12, max_w + 4)

    for _ in range(TEMPLATE_ATTEMPTS):
        th = random.randint(min_h, 18)
        tw = random.randint(min_w, 18)
        blocked = np.zeros((th, tw), dtype=bool)
        hole_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        ok = True
        dims_order = sorted(hole_dims, key=lambda dim: dim[0] * dim[1], reverse=True)
        for h, w in dims_order:
            candidates: list[tuple[int, int]] = []
            for r in range(1, th - h):
                for c in range(1, tw - w):
                    rr0 = max(0, r - 1)
                    rr1 = min(th, r + h + 1)
                    cc0 = max(0, c - 1)
                    cc1 = min(tw, c + w + 1)
                    if not blocked[rr0:rr1, cc0:cc1].any():
                        candidates.append((r, c))
            if not candidates:
                ok = False
                break
            r, c = random.choice(candidates)
            hole_boxes[(h, w)] = (r, c, r + h - 1, c + w - 1)
            _mark_blocked(blocked, r, c, h, w, pad=1)
        if not ok or len(hole_boxes) != len(hole_dims):
            continue

        patch = np.full((th, tw), BLUE, dtype=int)
        for (h, w), (r0, c0, r1, c1) in hole_boxes.items():
            patch[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        return patch, hole_boxes
    return None


def _place_source_sprites(
    shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    tr0, tc0, tr1, tc1 = template_bbox
    _mark_blocked(blocked, tr0, tc0, tr1 - tr0 + 1, tc1 - tc0 + 1, pad=1)

    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
        reverse=True,
    )
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                rr0 = max(0, r - 1)
                rr1 = min(h, r + ph + 1)
                cc0 = max(0, c - 1)
                cc1 = min(w, c + pw + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        positions[idx] = (r, c)
        _mark_blocked(blocked, r, c, ph, pw, pad=1)
    return [pos for pos in positions if pos is not None]


def _build_expected_output(
    grid_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    template_patch: np.ndarray,
    global_holes: dict[tuple[int, int], tuple[int, int, int, int]],
    sprite_meta: list[dict],
    *,
    use_processed: bool,
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = template_patch
    for meta in sprite_meta:
        bbox = global_holes[meta["shape"]]
        r0, c0, r1, c1 = bbox
        patch = meta["processed"] if use_processed else meta["raw"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    global_holes: dict[tuple[int, int], tuple[int, int, int, int]],
    sprite_meta: list[dict],
) -> bool:
    template_coords = max(_components_mask(inp == BLUE, connectivity=4), key=len)
    if _bbox_coords(template_coords) != template_bbox:
        return False

    template_patch = _crop_bbox(inp, template_bbox)
    hole_dims_found: list[tuple[int, int]] = []
    for coords in _components_mask(template_patch == BLACK, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if len(coords) != hh * ww:
            return False
        hole_dims_found.append((hh, ww))
    if sorted(hole_dims_found) != sorted(global_holes):
        return False

    all_sprite_components = 0
    for meta in sprite_meta:
        color = meta["color"]
        comps = _components_mask(inp == color, connectivity=4)
        if len(comps) != 1:
            return False
        all_sprite_components += 1
        bbox = _bbox_coords(comps[0])
        patch = _crop_bbox(inp, bbox)
        if patch.shape != meta["shape"]:
            return False
        if not np.array_equal(patch, meta["raw"]):
            return False
        axis, v_score, h_score = _best_axis(patch == color)
        if axis != meta["axis"]:
            return False
        if abs(v_score - h_score) < 2:
            return False
        if _detect_defects(patch == color, axis) != meta["defects"]:
            return False
        if not np.array_equal(_process_sprite_patch(patch, color, axis), meta["processed"]):
            return False
        if np.count_nonzero(meta["processed"] == ORANGE) == 0:
            return False
    if all_sprite_components != len(sprite_meta):
        return False

    naive = _build_expected_output(
        inp.shape,
        template_bbox,
        _crop_bbox(expected, template_bbox),
        global_holes,
        sprite_meta,
        use_processed=False,
    )
    if np.array_equal(naive, expected):
        return False

    return np.array_equal(main(inp), expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair near-symmetric source sprites, halo their missing cells, and embed them into the matching template cavities.

    Input structure:
    - A black background contains one large blue template component near the center.
    - Inside the template bbox are 2-3 disjoint rectangular black cavities.
    - Outside the template are 2-3 monochrome 4-connected sprites whose bbox sizes match the cavities.

    Rule:
    - Detect the blue template and the outside sprites, then read each sprite inside its own bbox.
    - For each sprite, choose the better of vertical or horizontal mirror symmetry by mirrored occupancy score.
    - Any black bbox cell whose mirror partner is the sprite color is a defect; mark the defect's in-bbox 8-neighbor black cells orange, then fill the defect itself with the sprite color.
    - Rebuild the output on the original canvas using only the blue template, and paste each repaired sprite bbox into the uniquely matching cavity with the same dimensions.

    Key insight:
    - The same asymmetric missing cells drive two coupled actions at once: they determine where the orange halo goes and which sprite pixels must be restored before size-based cavity placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue_comps = _components_mask(g == BLUE, connectivity=4)
    assert blue_comps, "expected a blue template"
    template_coords = max(blue_comps, key=len)
    template_bbox = _bbox_coords(template_coords)
    template_patch = _crop_bbox(g, template_bbox)

    hole_map: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    tr0, tc0, _tr1, _tc1 = template_bbox
    for coords in _components_mask(template_patch == BLACK, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert len(coords) == h * w, "cavities must be rectangular"
        key = (h, w)
        assert key not in hole_map, "cavity sizes must be unique"
        hole_map[key] = (tr0 + r0, tc0 + c0, tr0 + r1, tc0 + c1)

    out = np.zeros_like(g)
    out[template_coords[:, 0], template_coords[:, 1]] = BLUE

    sprite_infos: list[tuple[tuple[int, int], np.ndarray]] = []
    for color in sorted(set(int(x) for x in np.unique(g)) - {BLACK, BLUE}):
        for coords in _components_mask(g == color, connectivity=4):
            bbox = _bbox_coords(coords)
            patch = _crop_bbox(g, bbox)
            axis, _v_score, _h_score = _best_axis(patch == color)
            processed = _process_sprite_patch(patch, color, axis)
            sprite_infos.append((tuple(int(x) for x in patch.shape), processed))

    assert len(sprite_infos) == len(hole_map), "sprite and cavity counts must match"
    seen_dims: set[tuple[int, int]] = set()
    for shape, patch in sprite_infos:
        assert shape in hole_map, "every sprite must match a cavity"
        assert shape not in seen_dims, "sprite sizes must be unique"
        seen_dims.add(shape)
        r0, c0, r1, c1 = hole_map[shape]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        sprite_count = random.choice([2, 3])
        dims = random.sample(
            [(h, w) for h in range(4, 7) for w in range(4, 7)],
            sprite_count,
        )
        colors = random.sample(SPRITE_COLORS, sprite_count)

        sprite_meta: list[dict] = []
        ok = True
        for (h, w), color in zip(dims, colors):
            made = _make_sprite(h, w, color)
            if made is None:
                ok = False
                break
            _raw, _processed, meta = made
            sprite_meta.append(meta)
        if not ok:
            continue

        template_made = _make_template_patch([meta["shape"] for meta in sprite_meta])
        if template_made is None:
            continue
        template_patch, hole_boxes = template_made
        th, tw = template_patch.shape

        grid_h_min = max(24, th + 12)
        grid_w_min = max(24, tw + 12)
        if grid_h_min > 30 or grid_w_min > 30:
            continue
        grid_h = random.randint(grid_h_min, 30)
        grid_w = random.randint(grid_w_min, 30)

        tr0 = (grid_h - th) // 2
        tc0 = (grid_w - tw) // 2
        template_bbox = (tr0, tc0, tr0 + th - 1, tc0 + tw - 1)

        positions = _place_source_sprites(
            (grid_h, grid_w),
            template_bbox,
            [meta["raw"] for meta in sprite_meta],
        )
        if positions is None:
            continue

        inp = np.zeros((grid_h, grid_w), dtype=int)
        inp[tr0 : tr0 + th, tc0 : tc0 + tw] = template_patch
        for meta, (r0, c0) in zip(sprite_meta, positions):
            h, w = meta["shape"]
            inp[r0 : r0 + h, c0 : c0 + w] = meta["raw"]

        global_holes = {
            dim: (tr0 + r0, tc0 + c0, tr0 + r1, tc0 + c1)
            for dim, (r0, c0, r1, c1) in hole_boxes.items()
        }
        expected = _build_expected_output(
            inp.shape,
            template_bbox,
            template_patch,
            global_holes,
            sprite_meta,
            use_processed=True,
        )

        if _validate_case(inp, expected, template_bbox, global_holes, sprite_meta):
            return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
