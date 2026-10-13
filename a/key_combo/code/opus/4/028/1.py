import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SEED_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, TEAL, MAROON)
OPEN_SIDES = ("top", "bottom", "left", "right")
THICKNESS_TO_ROT = {1: 0, 2: -1, 3: 2}  # np.rot90 k values; -1 = 90 deg clockwise

TEMPLATE_MASKS = (
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
)

GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _full_rows_from_top(mask: np.ndarray) -> int:
    k = 0
    while k < mask.shape[0] and np.all(mask[k, :]):
        k += 1
    return int(k)


def _full_rows_from_bottom(mask: np.ndarray) -> int:
    k = 0
    while k < mask.shape[0] and np.all(mask[-1 - k, :]):
        k += 1
    return int(k)


def _full_cols_from_left(mask: np.ndarray) -> int:
    k = 0
    while k < mask.shape[1] and np.all(mask[:, k]):
        k += 1
    return int(k)


def _full_cols_from_right(mask: np.ndarray) -> int:
    k = 0
    while k < mask.shape[1] and np.all(mask[:, -1 - k]):
        k += 1
    return int(k)


def _detect_wall(mask: np.ndarray) -> tuple[int, str]:
    top = _full_rows_from_top(mask)
    bottom = _full_rows_from_bottom(mask)
    left = _full_cols_from_left(mask)
    right = _full_cols_from_right(mask)
    side_layers = {"top": top, "bottom": bottom, "left": left, "right": right}
    open_sides = [side for side, layers in side_layers.items() if layers == 0]
    assert len(open_sides) == 1
    opening = open_sides[0]
    t = max(side_layers.values())
    assert t in THICKNESS_TO_ROT
    non_zero = [layers for layers in side_layers.values() if layers > 0]
    assert non_zero and all(layers == t for layers in non_zero)
    return t, opening


def _cavity_slice(mask: np.ndarray, opening: str, t: int) -> tuple[slice, slice]:
    h, w = mask.shape
    if opening == "top":
        return slice(0, h - t), slice(t, w - t)
    if opening == "bottom":
        return slice(t, h), slice(t, w - t)
    if opening == "left":
        return slice(t, h - t), slice(0, w - t)
    return slice(t, h - t), slice(t, w)


def _rotate_template(template: np.ndarray, t: int) -> np.ndarray:
    return np.rot90(template, k=THICKNESS_TO_ROT[t])


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    view = dst[r0 : r0 + ph, c0 : c0 + pw]
    view[patch == ORANGE] = ORANGE


def _fill_black_components(region: np.ndarray, color: int) -> None:
    black_comps = _components(region == BLACK)
    for comp in black_comps:
        region[comp[:, 0], comp[:, 1]] = int(color)


def _extract_template(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid == ORANGE)
    assert len(comps) == 1
    box = _bbox(comps[0])
    template = _crop(grid, box)
    assert template.shape == (3, 3)
    return template


def _wall_specs_from_grid(grid: np.ndarray):
    specs = []
    for coords in _components(grid == GREY):
        box = _bbox(coords)
        mask = _crop(grid == GREY, box)
        t, opening = _detect_wall(mask)
        rs, cs = _cavity_slice(mask, opening, t)
        cavity = grid[box[0] + rs.start : box[0] + rs.stop, box[1] + cs.start : box[1] + cs.stop]
        seed_coords = np.argwhere((cavity != BLACK) & (cavity != GREY) & (cavity != ORANGE))
        assert seed_coords.shape == (1, 2)
        sr, sc = map(int, seed_coords[0])
        seed_color = int(cavity[sr, sc])
        specs.append(
            {
                "bbox": box,
                "mask": mask,
                "thickness": t,
                "opening": opening,
                "seed": (int(box[0] + rs.start + sr), int(box[1] + cs.start + sc)),
                "seed_color": seed_color,
                "cavity_box": (
                    int(box[0] + rs.start),
                    int(box[1] + cs.start),
                    int(box[0] + rs.stop - 1),
                    int(box[1] + cs.stop - 1),
                ),
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the outside orange template by wall thickness and inlay it into each semi-enclosed chamber, then flood-fill the remaining cavity.

    Input structure:
    - The background is black, with one orange 3x3 asymmetric template placed in a corner.
    - Several grey 4-connected three-sided walls appear elsewhere; each wall is an axis-aligned U/C shape with thickness 1, 2, or 3.
    - Each wall encloses one rectangular cavity, and that cavity contains exactly one non-black, non-grey seed pixel.

    Rule:
    - For each grey wall, measure its thickness by counting consecutive full grey layers on the closed sides.
    - Map thickness 1/2/3 to template rotations 0 deg / 90 deg clockwise / 180 deg, then place the rotated orange template so the seed pixel is the rotated template's top-left corner.
    - Inside that cavity, recolor every remaining black region to the seed color; walls, background, and the original corner template stay unchanged.

    Key insight:
    - The wall thickness is the hidden rotation code, while the seed pixel fixes the inlay anchor, so both geometric cues are needed before the cavity can be filled.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    template = _extract_template(g)

    for spec in _wall_specs_from_grid(g):
        seed_r, seed_c = spec["seed"]
        patch = _rotate_template(template, spec["thickness"])
        _paint_patch(out, patch, seed_r, seed_c)

        r0, c0, r1, c1 = spec["cavity_box"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        _fill_black_components(region, spec["seed_color"])

    return out


def _template_ok(mask: np.ndarray) -> bool:
    if mask.shape != (3, 3):
        return False
    if len(_components(mask.astype(bool))) != 1:
        return False
    rots = [np.rot90(mask, k=k) for k in (0, -1, 2)]
    keys = {rot.tobytes() for rot in rots}
    if len(keys) != 3:
        return False
    covers = [bool(rot[0, 0]) for rot in rots]
    return any(covers) and not all(covers)


def _wall_mask(opening: str, t: int, inner_h: int, inner_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    if opening in ("top", "bottom"):
        h = inner_h + t
        w = inner_w + 2 * t
    else:
        h = inner_h + 2 * t
        w = inner_w + t

    mask = np.zeros((h, w), dtype=bool)
    if opening == "top":
        mask[:, :t] = True
        mask[:, w - t :] = True
        mask[h - t :, :] = True
        cavity_origin = (0, t)
    elif opening == "bottom":
        mask[:, :t] = True
        mask[:, w - t :] = True
        mask[:t, :] = True
        cavity_origin = (t, t)
    elif opening == "left":
        mask[:t, :] = True
        mask[h - t :, :] = True
        mask[:, w - t :] = True
        cavity_origin = (t, 0)
    else:
        mask[:t, :] = True
        mask[h - t :, :] = True
        mask[:, :t] = True
        cavity_origin = (t, t)
    return mask, cavity_origin


def _corner_position(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, w - 3
    if corner == "bottom_left":
        return h - 3, 0
    return h - 3, w - 3


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _try_place_bbox(reserved: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    candidates = []
    for r0 in range(1, reserved.shape[0] - h):
        for c0 in range(1, reserved.shape[1] - w):
            if reserved[r0 : r0 + h, c0 : c0 + w].any():
                continue
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + h + 1
            cc1 = c0 + w + 1
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full((spec["H"], spec["W"]), BLACK, dtype=int)

    tr, tc = spec["template_pos"]
    template_patch = np.where(spec["template_mask"], ORANGE, BLACK)
    grid[tr : tr + 3, tc : tc + 3] = template_patch

    for wall in spec["walls"]:
        r0, c0 = wall["bbox"][:2]
        h, w = wall["mask"].shape
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        sub[wall["mask"]] = GREY
        sr, sc = wall["seed"]
        grid[sr, sc] = wall["seed_color"]
    return grid


def _expected_output_from_spec(spec: dict) -> np.ndarray:
    out = _render_input(spec)
    template_patch = np.where(spec["template_mask"], ORANGE, BLACK)
    for wall in spec["walls"]:
        seed_r, seed_c = wall["seed"]
        patch = _rotate_template(template_patch, wall["thickness"])
        _paint_patch(out, patch, seed_r, seed_c)

        r0, c0, r1, c1 = wall["cavity_box"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == BLACK] = wall["seed_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    num_walls = random.choice((3, 4))
    H = random.randint(24 if num_walls == 4 else 22, 29)
    W = random.randint(24 if num_walls == 4 else 22, 29)

    template_mask = random.choice(TEMPLATE_MASKS)
    if not _template_ok(template_mask):
        return None

    thicknesses = [1, 2, 3]
    if num_walls == 4:
        thicknesses.append(random.choice((1, 2, 3)))
    random.shuffle(thicknesses)

    if num_walls == 4:
        openings = list(OPEN_SIDES)
        random.shuffle(openings)
    else:
        openings = random.sample(OPEN_SIDES, 3)

    seed_colors = random.sample(SEED_COLORS, num_walls)
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    template_pos = _corner_position(H, W, corner)

    reserved = np.zeros((H, W), dtype=bool)
    _mark_reserved(reserved, template_pos[0], template_pos[1], 3, 3, pad=2)

    walls = []
    for idx in range(num_walls):
        t = thicknesses[idx]
        opening = openings[idx]
        inner_h = random.randint(5, 6)
        inner_w = random.randint(5, 6)
        mask, cavity_origin = _wall_mask(opening, t, inner_h, inner_w)
        pos = _try_place_bbox(reserved, mask.shape[0], mask.shape[1])
        if pos is None:
            return None

        r0, c0 = pos
        _mark_reserved(reserved, r0, c0, mask.shape[0], mask.shape[1], pad=1)

        seed_r = r0 + cavity_origin[0] + random.randint(0, inner_h - 3)
        seed_c = c0 + cavity_origin[1] + random.randint(0, inner_w - 3)
        cavity_box = (
            r0 + cavity_origin[0],
            c0 + cavity_origin[1],
            r0 + cavity_origin[0] + inner_h - 1,
            c0 + cavity_origin[1] + inner_w - 1,
        )
        walls.append(
            {
                "bbox": (r0, c0, r0 + mask.shape[0] - 1, c0 + mask.shape[1] - 1),
                "mask": mask,
                "opening": opening,
                "thickness": t,
                "seed": (seed_r, seed_c),
                "seed_color": seed_colors[idx],
                "cavity_box": cavity_box,
            }
        )

    spec = {"H": H, "W": W, "template_mask": template_mask, "template_pos": template_pos, "walls": walls}
    inp = _render_input(spec)
    expected = _expected_output_from_spec(spec)
    actual = main(inp)

    if not np.array_equal(actual, expected):
        return None

    counts = Counter(wall["thickness"] for wall in walls)
    if any(counts[t] < 1 for t in (1, 2, 3)):
        return None
    if len({wall["opening"] for wall in walls}) < min(3, num_walls):
        return None

    for wall in walls:
        r0, c0, r1, c1 = wall["cavity_box"]
        cavity_out = actual[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(cavity_out == ORANGE):
            return None
        if int((cavity_out == wall["seed_color"]).sum()) == 0:
            return None
        if np.any(cavity_out == BLACK):
            return None

    if len(_components(inp == GREY)) != num_walls:
        return None
    if len(_components(inp == ORANGE)) != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid wall-thickness rotation puzzle")
