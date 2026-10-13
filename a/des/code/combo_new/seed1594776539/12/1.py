from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500
ZONES = ("tl", "tr", "bl", "br")


def _bool_mask(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _bool_mask("10", "10", "11"),
    _bool_mask("110", "011", "010"),
    _bool_mask("011", "110", "010"),
    _bool_mask("111", "100", "100"),
    _bool_mask("111", "010", "010"),
    _bool_mask("101", "111"),
    _bool_mask("1100", "0100", "0111"),
    _bool_mask("1110", "0010", "0111"),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_mask(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return r0, c0, mask


def _rotation_variants(arr: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(arr, k) for k in range(4)]


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(rot.shape), rot.tobytes()) for rot in _rotation_variants(mask.astype(bool)))


def _validate_shape_library() -> None:
    keys = set()
    for mask in SHAPE_LIBRARY:
        variants = {(tuple(rot.shape), rot.tobytes()) for rot in _rotation_variants(mask)}
        assert len(variants) == 4, "every generator shape must have four distinct rotations"
        key = _rotation_key(mask)
        assert key not in keys, "shape library must contain unique rotation classes"
        keys.add(key)


_validate_shape_library()


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_pts = np.argwhere(grid == GREY)
    assert len(grey_pts) >= 8, "frame must be present"
    r0, c0, r1, c1 = _bbox(grey_pts)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"
    assert np.all(grid[r0, c0 : c1 + 1] == GREY)
    assert np.all(grid[r1, c0 : c1 + 1] == GREY)
    assert np.all(grid[r0 : r1 + 1, c0] == GREY)
    assert np.all(grid[r0 : r1 + 1, c1] == GREY)
    return r0, c0, r1, c1


def _extract_holes(interior: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    holes: list[tuple[int, int, np.ndarray]] = []
    for coords in _components(interior == BLACK):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == interior.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == interior.shape[1] - 1):
            continue
        holes.append(_component_mask(coords))
    holes.sort(key=lambda item: (item[0], item[1]))
    return holes


def _extract_templates(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.array(grid, copy=True)
    outside[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    templates: list[tuple[int, int, np.ndarray]] = []
    for coords in _components(outside != BLACK):
        tr0, tc0, tr1, tc1 = _bbox(coords)
        templates.append((tr0, tc0, outside[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()))
    templates.sort(key=lambda item: (item[0], item[1]))
    return [patch for _, _, patch in templates]


def _paint_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black gap inside the grey frame with the matching outside template after rotating it to the required orientation.

    Input structure:
    - The background is black.
    - A single grey 1-pixel rectangular frame encloses several monochrome connected blocks, with black space both around them and inside some of them.
    - Outside the frame are several compact multi-color template objects; each template has a unique overall silhouette.

    Rule:
    - Find the grey frame and inspect only its interior.
    - Among the interior black cells, ignore the border-connected black region and keep only the enclosed 4-connected black components as missing holes.
    - For each hole, find the unique outside template whose non-black silhouette matches the hole after a rotation by 0, 90, 180, or 270 degrees.
    - Rotate that full multi-color template by the same amount and paint it into the hole.
    - Remove all outside templates by turning the area outside the frame black, while keeping the frame and the pre-existing interior blocks unchanged.

    Key insight:
    The hole shape identifies both which template to use and how much to rotate it; the template's internal colors are transferred only after this geometric match is resolved.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _find_frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    holes = _extract_holes(interior)
    templates = _extract_templates(g, frame_bbox)
    assert holes and len(holes) == len(templates), "hole/template count mismatch"

    out = g.copy()
    outside_mask = np.ones(g.shape, dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    out[outside_mask] = BLACK

    used: set[int] = set()
    for hr0, hc0, hole_mask in holes:
        matches: list[tuple[int, int]] = []
        for idx, patch in enumerate(templates):
            if idx in used:
                continue
            tpl_mask = patch != BLACK
            for k in range(4):
                rot_mask = np.rot90(tpl_mask, k)
                if rot_mask.shape == hole_mask.shape and np.array_equal(rot_mask, hole_mask):
                    matches.append((idx, k))
        assert len(matches) == 1, "each hole must match exactly one template and rotation"
        tpl_idx, rot_k = matches[0]
        used.add(tpl_idx)
        rotated_patch = np.rot90(templates[tpl_idx], rot_k)
        _paint_nonzero(out, rotated_patch, r0 + 1 + hr0, c0 + 1 + hc0)

    return out


def _zone_bounds(inner_h: int, inner_w: int, split_r: int, split_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (1, split_r - 1, 1, split_c - 1),
        "tr": (1, split_r - 1, split_c + 1, inner_w - 2),
        "bl": (split_r + 1, inner_h - 2, 1, split_c - 1),
        "br": (split_r + 1, inner_h - 2, split_c + 1, inner_w - 2),
    }


def _randint_pair(low: int, high: int) -> int:
    assert low <= high
    return random.randint(low, high)


def _make_template_patch(mask: np.ndarray, forbidden: set[int]) -> np.ndarray:
    cells = [tuple(x) for x in np.argwhere(mask)]
    avail = [c for c in COLOR_POOL if c not in forbidden]
    assert len(avail) >= 3
    for _ in range(120):
        n_colors = 2 if len(cells) <= 4 else random.choice((2, 3))
        colors = random.sample(avail, n_colors)
        assigned = colors + [random.choice(colors) for _ in range(len(cells) - n_colors)]
        random.shuffle(assigned)
        patch = np.zeros(mask.shape, dtype=int)
        for (r, c), color in zip(cells, assigned):
            patch[r, c] = color
        if len(set(int(v) for v in patch[mask])) != n_colors:
            continue
        variants = {(tuple(rot.shape), rot.tobytes()) for rot in _rotation_variants(patch)}
        if len(variants) == 4:
            return patch
    raise RuntimeError("failed to create a non-symmetric colored template")


def _place_template_outside(grid: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, gh - h):
        for c0 in range(1, gw - w):
            window = grid[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]
            if np.all(window == BLACK):
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    _paint_nonzero(grid, patch, r0, c0)
    return True


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_holes = random.randint(2, 4)
    shape_indices = random.sample(range(len(SHAPE_LIBRARY)), n_holes)
    zones = random.sample(list(ZONES), n_holes)
    block_colors = random.sample(COLOR_POOL, n_holes)

    hole_specs: list[dict[str, object]] = []
    for shape_idx, zone, block_color in zip(shape_indices, zones, block_colors):
        base_mask = SHAPE_LIBRARY[shape_idx]
        base_patch = _make_template_patch(base_mask, {int(block_color)})
        outside_k = random.randrange(4)
        outside_patch = np.rot90(base_patch, outside_k)
        fill_k = random.randrange(4)
        filled_patch = np.rot90(outside_patch, fill_k)
        hole_mask = filled_patch != BLACK
        hole_h, hole_w = hole_mask.shape
        rect_h = hole_h + random.randint(2, 4)
        rect_w = hole_w + random.randint(2, 4)
        hole_specs.append(
            {
                "zone": zone,
                "block_color": int(block_color),
                "outside_patch": outside_patch,
                "filled_patch": filled_patch,
                "hole_mask": hole_mask,
                "rect_h": int(rect_h),
                "rect_w": int(rect_w),
            }
        )

    top_need = max(int(spec["rect_h"]) for spec in hole_specs if spec["zone"] in ("tl", "tr")) if any(spec["zone"] in ("tl", "tr") for spec in hole_specs) else 2
    bot_need = max(int(spec["rect_h"]) for spec in hole_specs if spec["zone"] in ("bl", "br")) if any(spec["zone"] in ("bl", "br") for spec in hole_specs) else 2
    left_need = max(int(spec["rect_w"]) for spec in hole_specs if spec["zone"] in ("tl", "bl")) if any(spec["zone"] in ("tl", "bl") for spec in hole_specs) else 2
    right_need = max(int(spec["rect_w"]) for spec in hole_specs if spec["zone"] in ("tr", "br")) if any(spec["zone"] in ("tr", "br") for spec in hole_specs) else 2

    inner_h_min = max(10, top_need + bot_need + 3)
    inner_w_min = max(10, left_need + right_need + 3)
    inner_h_max = 18
    inner_w_max = 18
    if inner_h_min > inner_h_max or inner_w_min > inner_w_max:
        return None

    inner_h = random.randint(inner_h_min, inner_h_max)
    inner_w = random.randint(inner_w_min, inner_w_max)
    split_r = _randint_pair(1 + top_need, inner_h - 2 - bot_need)
    split_c = _randint_pair(1 + left_need, inner_w - 2 - right_need)
    zone_bounds = _zone_bounds(inner_h, inner_w, split_r, split_c)

    frame_h, frame_w = inner_h + 2, inner_w + 2
    grid_h_min = max(20, frame_h + 8)
    grid_w_min = max(20, frame_w + 8)
    if grid_h_min > 30 or grid_w_min > 30:
        return None
    grid_h = random.randint(grid_h_min, 30)
    grid_w = random.randint(grid_w_min, 30)
    frame_r0 = random.randint(2, grid_h - frame_h - 2)
    frame_c0 = random.randint(2, grid_w - frame_w - 2)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    exp = np.zeros((grid_h, grid_w), dtype=int)
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1
    inp[frame_r0, frame_c0 : frame_c1 + 1] = GREY
    inp[frame_r1, frame_c0 : frame_c1 + 1] = GREY
    inp[frame_r0 : frame_r1 + 1, frame_c0] = GREY
    inp[frame_r0 : frame_r1 + 1, frame_c1] = GREY
    exp[:, :] = inp

    for spec in hole_specs:
        zr0, zr1, zc0, zc1 = zone_bounds[str(spec["zone"])]
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        zone_h = zr1 - zr0 + 1
        zone_w = zc1 - zc0 + 1
        if rect_h > zone_h or rect_w > zone_w:
            return None
        rect_r = random.randint(zr0, zr1 - rect_h + 1)
        rect_c = random.randint(zc0, zc1 - rect_w + 1)

        abs_r = frame_r0 + 1 + rect_r
        abs_c = frame_c0 + 1 + rect_c
        inp[abs_r : abs_r + rect_h, abs_c : abs_c + rect_w] = int(spec["block_color"])
        exp[abs_r : abs_r + rect_h, abs_c : abs_c + rect_w] = int(spec["block_color"])

        hole_mask = np.asarray(spec["hole_mask"], dtype=bool)
        hole_h, hole_w = hole_mask.shape
        hole_r = rect_r + random.randint(1, rect_h - hole_h - 1)
        hole_c = rect_c + random.randint(1, rect_w - hole_w - 1)
        hole_abs_r = frame_r0 + 1 + hole_r
        hole_abs_c = frame_c0 + 1 + hole_c
        hole_view = inp[hole_abs_r : hole_abs_r + hole_h, hole_abs_c : hole_abs_c + hole_w]
        hole_view[hole_mask] = BLACK
        _paint_nonzero(exp, np.asarray(spec["filled_patch"], dtype=int), hole_abs_r, hole_abs_c)

    holes = _extract_holes(inp[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1])
    if len(holes) != n_holes:
        return None

    for spec in hole_specs:
        if not _place_template_outside(inp, np.asarray(spec["outside_patch"], dtype=int)):
            return None

    if len(_extract_templates(inp, (frame_r0, frame_c0, frame_r1, frame_c1))) != n_holes:
        return None

    return inp, exp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, exp = built
        try:
            if np.array_equal(main(inp), exp):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
