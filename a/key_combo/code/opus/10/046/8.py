from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLORS = (5, 8, 9)
GEN_ATTEMPTS = 500
GRID_MIN = 16
GRID_MAX = 20
INTERIOR_MIN = 4
INTERIOR_MAX = 7
SPRITE_AREAS = (3, 4, 5, 6)

PATTERNS: dict[int, list[tuple[str, np.ndarray]]] = {
    3: [
        ("open", np.array([[1, 1], [1, 0]], dtype=bool)),
    ],
    4: [
        ("closed", np.array([[1, 1], [1, 1]], dtype=bool)),
        ("open", np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)),
        ("open", np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)),
        ("open", np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)),
    ],
    5: [
        ("open", np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)),
        ("open", np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)),
        ("open", np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)),
        ("open", np.array([[1, 1], [1, 1], [1, 0]], dtype=bool)),
    ],
    6: [
        ("closed", np.array([[1, 1, 1], [1, 1, 1]], dtype=bool)),
        ("open", np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool)),
        ("open", np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rect_frame_bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(local)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(local & ~border):
        return None
    perimeter = int(border.sum())
    if int(local.sum()) < perimeter - 10:
        return None
    if not (local[0].any() and local[-1].any() and local[:, 0].any() and local[:, -1].any()):
        return None
    return r0, c0, r1, c1


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if src_len == 1:
        assert dst_len == 1, "single-pixel source axis requires single-pixel target axis"
        return 0
    assert dst_len >= 2
    return int(round(i * (dst_len - 1) / (src_len - 1)))


def _draw_line(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        c_start, c_stop = sorted((c0, c1))
        mask[r0, c_start : c_stop + 1] = True
        return
    if c0 == c1:
        r_start, r_stop = sorted((r0, r1))
        mask[r_start : r_stop + 1, c0] = True
        return
    raise AssertionError("expected axis-aligned segment")


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke.astype(bool)
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    coords = np.argwhere(sprite)
    assert len(coords) > 0

    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in coords:
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = True

    for sr, sc in coords:
        sr = int(sr)
        sc = int(sc)
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in mapped:
                r0, c0 = mapped[(sr, sc)]
                r1, c1 = mapped[(nr, nc)]
                _draw_line(stroke, r0, c0, r1, c1)

    return _fill_enclosed(stroke)


def _render(shape: tuple[int, int], visible_frames: np.ndarray, stretched_full: dict[int, np.ndarray], order: tuple[int, ...]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        out[stretched_full[color]] = int(color)
    frame_mask = visible_frames != BG
    out[frame_mask] = visible_frames[frame_mask]
    return out


def _identify_color_parts(grid: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    comps = _component_coords(grid == color)
    candidates: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = []
    for idx, sprite_coords in enumerate(comps):
        frame_parts = [coords for j, coords in enumerate(comps) if j != idx]
        if not frame_parts:
            continue
        frame_coords = np.concatenate(frame_parts, axis=0)
        frame_bbox = _rect_frame_bbox_from_coords(frame_coords)
        if frame_bbox is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bbox
        sr0, sc0, sr1, sc1 = _bbox_coords(sprite_coords)
        outside = sr1 < fr0 or sr0 > fr1 or sc1 < fc0 or sc0 > fc1
        if not outside:
            continue
        candidates.append((sprite_coords, frame_coords, frame_bbox))

    assert len(candidates) == 1, "expected a unique sprite/frame split per color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Stretch each same-colored sprite into its matching hollow frame and resolve overlaps by sprite area.

    Input structure:
    - The background is black.
    - Exactly three colors each appear twice as two 4-connected components: a
      one-pixel-thick hollow rectangular frame and a small sprite outside the frames.
    - Frame interiors overlap in space, while the three source sprites are separated
      from one another in one corner region.

    Rule:
    - Pair each sprite with the frame of the same color.
    - Map the sprite bbox to the matching frame interior bbox, send bbox corners to
      interior corners, connect mapped 4-neighbor sprite pixels with orthogonal
      segments, then fill enclosed regions.
    - Draw the three stretched results from largest source sprite area to smallest,
      so smaller sprites are rendered later and occlude earlier colors in overlaps.
    - Preserve every frame border pixel and clear the original sprite locations.

    Key insight:
    The difficult part is not the stretching alone: overlapping target interiors make
    the final ownership of shared pixels depend on sorting the source sprites by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    assert bg == BG

    frame_specs: dict[int, dict] = {}
    sprite_specs: dict[int, dict] = {}
    for color in _colors_present(g, exclude={bg}):
        sprite_part, frame_coords, frame_bbox = _identify_color_parts(g, color)
        sr0, sc0, sr1, sc1 = _bbox_coords(sprite_part)
        sprite_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        sprite_mask[sprite_part[:, 0] - sr0, sprite_part[:, 1] - sc0] = True

        fr0, fc0, fr1, fc1 = frame_bbox
        ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
        assert ir0 <= ir1 and ic0 <= ic1

        frame_specs[color] = {
            "bbox": frame_bbox,
            "interior_bbox": (ir0, ic0, ir1, ic1),
            "coords": frame_coords,
        }
        sprite_specs[color] = {
            "bbox": (sr0, sc0, sr1, sc1),
            "mask": sprite_mask,
            "area": int(len(sprite_part)),
            "coords": sprite_part,
        }

    assert set(frame_specs) == set(FRAME_COLORS), "expected exactly the three frame colors"

    areas = [sprite_specs[color]["area"] for color in FRAME_COLORS]
    assert len(set(areas)) == 3, "sprite areas must be pairwise distinct"

    stretched_full: dict[int, np.ndarray] = {}
    visible_frames = g.copy()
    for color in FRAME_COLORS:
        visible_frames[sprite_specs[color]["coords"][:, 0], sprite_specs[color]["coords"][:, 1]] = BG
    for color in FRAME_COLORS:
        ir0, ic0, ir1, ic1 = frame_specs[color]["interior_bbox"]
        interior_shape = (ir1 - ir0 + 1, ic1 - ic0 + 1)
        local = _stretch_sprite(sprite_specs[color]["mask"], interior_shape)
        full = np.zeros_like(g, dtype=bool)
        full[ir0 : ir1 + 1, ic0 : ic1 + 1] = local
        stretched_full[color] = full

    order = tuple(sorted(FRAME_COLORS, key=lambda color: -sprite_specs[color]["area"]))
    return _render(g.shape, visible_frames, stretched_full, order)


def _random_pattern(area: int, *, require_closed: bool | None = None) -> np.ndarray:
    options = PATTERNS[area]
    if require_closed is not None:
        options = [item for item in options if (item[0] == "closed") == require_closed]
    _kind, mask = random.choice(options)
    out = mask.copy()
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return _crop_to_mask(out)


def _bbox_gap_ok(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros(shape, dtype=bool)
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    return out


def _interior_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros(shape, dtype=bool)
    out[r0 + 1 : r1, c0 + 1 : c1] = True
    return out


def _sprite_zone_bbox(shape: tuple[int, int], corner: str, side: int = 7) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0, side - 1, side - 1
    if corner == "tr":
        return 0, w - side, side - 1, w - 1
    if corner == "bl":
        return h - side, 0, h - 1, side - 1
    return h - side, w - side, h - 1, w - 1


def _candidate_sprite_positions(shape: tuple[int, int], patch_shape: tuple[int, int], corner: str, side: int = 7) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch_shape
    if corner == "tl":
        rows = range(0, side - ph + 1)
        cols = range(0, side - pw + 1)
    elif corner == "tr":
        rows = range(0, side - ph + 1)
        cols = range(w - side, w - pw + 1)
    elif corner == "bl":
        rows = range(h - side, h - ph + 1)
        cols = range(0, side - pw + 1)
    else:
        rows = range(h - side, h - ph + 1)
        cols = range(w - side, w - pw + 1)
    return [(r, c) for r in rows for c in cols]


def _sample_sprite_specs() -> dict[int, dict]:
    colors = list(FRAME_COLORS)
    random.shuffle(colors)
    areas = random.sample(SPRITE_AREAS, 3)
    closed_color = random.choice([color for color, area in zip(colors, areas) if area in (4, 6)])

    specs: dict[int, dict] = {}
    for color, area in zip(colors, areas):
        require_closed = True if color == closed_color else None
        if require_closed and area not in (4, 6):
            require_closed = None
        mask = _random_pattern(area, require_closed=require_closed)
        specs[color] = {"area": area, "mask": mask}
    return specs


def _sample_frame_layout(shape: tuple[int, int]) -> dict[int, dict] | None:
    h, w = shape
    for _ in range(200):
        colors = list(FRAME_COLORS)
        random.shuffle(colors)

        dims: dict[int, tuple[int, int]] = {}
        used_areas = set()
        ok = True
        for color in colors:
            for _ in range(20):
                ih = random.randint(INTERIOR_MIN, INTERIOR_MAX)
                iw = random.randint(INTERIOR_MIN, INTERIOR_MAX)
                area = ih * iw
                if area in used_areas:
                    continue
                dims[color] = (ih, iw)
                used_areas.add(area)
                break
            else:
                ok = False
                break
        if not ok:
            continue

        anchor_r = random.randint(4, h - 5)
        anchor_c = random.randint(4, w - 5)
        specs: dict[int, dict] = {}
        for color in colors:
            ih, iw = dims[color]
            top_min = max(1, anchor_r - ih + 1)
            top_max = min(anchor_r, h - ih - 1)
            left_min = max(1, anchor_c - iw + 1)
            left_max = min(anchor_c, w - iw - 1)
            if top_min > top_max or left_min > left_max:
                break

            candidates = [(top, left) for top in range(top_min, top_max + 1) for left in range(left_min, left_max + 1)]
            random.shuffle(candidates)
            for top, left in candidates:
                specs[color] = {
                    "bbox": (top - 1, left - 1, top + ih, left + iw),
                    "interior_bbox": (top, left, top + ih - 1, left + iw - 1),
                    "interior_area": ih * iw,
                }
                break
            else:
                break
        else:
            interior_masks = {color: _interior_mask(shape, spec["bbox"]) for color, spec in specs.items()}
            if not all(np.any(interior_masks[a] & interior_masks[b]) for a, b in itertools.combinations(FRAME_COLORS, 2)):
                continue
            return specs
    return None


def _place_stretched_full(shape: tuple[int, int], frame_specs: dict[int, dict], sprite_specs: dict[int, dict]) -> dict[int, np.ndarray]:
    full_masks: dict[int, np.ndarray] = {}
    for color in FRAME_COLORS:
        ir0, ic0, ir1, ic1 = frame_specs[color]["interior_bbox"]
        local = _stretch_sprite(sprite_specs[color]["mask"], (ir1 - ir0 + 1, ic1 - ic0 + 1))
        full = np.zeros(shape, dtype=bool)
        full[ir0 : ir1 + 1, ic0 : ic1 + 1] = local
        full_masks[color] = full
    return full_masks


def _place_source_sprites(shape: tuple[int, int], corner: str, sprite_specs: dict[int, dict], frame_specs: dict[int, dict]) -> dict[int, tuple[int, int]] | None:
    frame_boxes = [spec["bbox"] for spec in frame_specs.values()]
    placed: dict[int, tuple[int, int]] = {}
    used_boxes: list[tuple[int, int, int, int]] = []
    colors = list(FRAME_COLORS)
    random.shuffle(colors)
    for color in colors:
        patch = sprite_specs[color]["mask"]
        candidates = _candidate_sprite_positions(shape, patch.shape, corner)
        random.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + patch.shape[0] - 1, c + patch.shape[1] - 1)
            if any(not _bbox_gap_ok(box, prev, gap=1) for prev in used_boxes):
                continue
            if any(not _bbox_gap_ok(box, frame_box, gap=0) for frame_box in frame_boxes):
                continue
            used_boxes.append(box)
            placed[color] = (r, c)
            break
        else:
            return None
    return placed


def _build_input(
    shape: tuple[int, int],
    frame_specs: dict[int, dict],
    sprite_specs: dict[int, dict],
    sprite_positions: dict[int, tuple[int, int]],
    frame_order: tuple[int, ...],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color in frame_order:
        spec = frame_specs[color]
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    for color, (r, c) in sprite_positions.items():
        patch = np.where(sprite_specs[color]["mask"], color, BG)
        h, w = patch.shape
        view = grid[r : r + h, c : c + w]
        assert np.all((view == BG) | (patch == BG))
        view[patch != BG] = color
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray, frame_specs: dict[int, dict], sprite_specs: dict[int, dict], stretched_full: dict[int, np.ndarray]) -> bool:
    intended_order = tuple(sorted(FRAME_COLORS, key=lambda color: -sprite_specs[color]["area"]))
    frame_order = tuple(sorted(FRAME_COLORS, key=lambda color: -frame_specs[color]["interior_area"]))
    numeric_order = tuple(sorted(FRAME_COLORS))
    if intended_order == frame_order or intended_order == numeric_order or intended_order == tuple(reversed(numeric_order)):
        return False

    overlap_counts = np.zeros(inp.shape, dtype=int)
    for color in FRAME_COLORS:
        overlap_counts += stretched_full[color].astype(int)
    if np.count_nonzero(overlap_counts >= 2) < 3:
        return False
    if np.count_nonzero(overlap_counts >= 2) > 28:
        return False

    visible_frames = inp.copy()
    for color in FRAME_COLORS:
        try:
            sprite_coords, _frame_coords, _frame_bbox = _identify_color_parts(inp, color)
        except AssertionError:
            return False
        visible_frames[sprite_coords[:, 0], sprite_coords[:, 1]] = BG

    rendered = {intended_order: _render(inp.shape, visible_frames, stretched_full, intended_order)}
    if not np.array_equal(rendered[intended_order], out):
        return False

    alt_orders = {
        tuple(reversed(intended_order)),
        frame_order,
        numeric_order,
        tuple(reversed(numeric_order)),
    }
    distinct = {rendered[intended_order].tobytes()}
    for alt_order in alt_orders:
        if alt_order == intended_order:
            continue
        alt = _render(inp.shape, visible_frames, stretched_full, alt_order)
        if np.array_equal(rendered[intended_order], alt):
            return False
        distinct.add(alt.tobytes())
    if len(distinct) < 3:
        return False

    border_union = visible_frames != BG
    for color in FRAME_COLORS:
        if not np.any((out == color) & ~border_union):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    frame_specs = _sample_frame_layout(shape)
    if frame_specs is None:
        return None

    sprite_specs = None
    stretched_full = None
    frame_order = tuple(sorted(FRAME_COLORS, key=lambda color: -frame_specs[color]["interior_area"]))
    numeric_order = tuple(sorted(FRAME_COLORS))
    for _ in range(30):
        trial_specs = _sample_sprite_specs()
        intended_order = tuple(sorted(FRAME_COLORS, key=lambda color: -trial_specs[color]["area"]))
        if intended_order == frame_order or intended_order == numeric_order or intended_order == tuple(reversed(numeric_order)):
            continue
        trial_masks = _place_stretched_full(shape, frame_specs, trial_specs)
        if not all(np.any(trial_masks[a] & trial_masks[b]) for a, b in itertools.combinations(FRAME_COLORS, 2)):
            continue
        sprite_specs = trial_specs
        stretched_full = trial_masks
        break
    if sprite_specs is None or stretched_full is None:
        return None

    corners = ["tl", "tr", "bl", "br"]
    random.shuffle(corners)
    sprite_positions = None
    for corner in corners:
        sprite_positions = _place_source_sprites(shape, corner, sprite_specs, frame_specs)
        if sprite_positions is not None:
            break
    if sprite_positions is None:
        return None

    order = tuple(sorted(FRAME_COLORS, key=lambda color: -sprite_specs[color]["area"]))
    frame_orders = list(itertools.permutations(FRAME_COLORS))
    random.shuffle(frame_orders)
    for frame_order in frame_orders:
        inp = _build_input(shape, frame_specs, sprite_specs, sprite_positions, frame_order)
        visible_frames = inp.copy()
        for color in FRAME_COLORS:
            patch = sprite_specs[color]["mask"]
            r, c = sprite_positions[color]
            visible_frames[r : r + patch.shape[0], c : c + patch.shape[1]][patch] = BG
        out = _render(shape, visible_frames, stretched_full, order)
        if not _semantic_checks(inp, out, frame_specs, sprite_specs, stretched_full):
            continue

        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
