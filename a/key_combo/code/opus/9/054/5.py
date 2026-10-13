from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BASE_COLOR = 1
CONNECTOR_COLOR = 2
CANDIDATE_COLORS = [3, 4, 5, 6, 7, 8, 9]

TYPE_FLIPLR = "horizontal_reflection"
TYPE_FLIPUD = "vertical_reflection"
TYPE_ROT180 = "rot180"

TYPE_TO_DELTA = {
    TYPE_FLIPLR: (0, 1),
    TYPE_FLIPUD: (1, 0),
    TYPE_ROT180: (-1, 0),
}

TYPE_TO_SIDES = {
    TYPE_FLIPLR: ("right", "left"),
    TYPE_FLIPUD: ("bottom", "top"),
    TYPE_ROT180: ("top", "bottom"),
}

FLIPLR_TEMPLATES = [
    (np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool), (0, 0)),
    (np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=bool), (0, 0)),
]

FLIPUD_TEMPLATES = [
    (mask.T.copy(), (0, 0)) for mask, _anchor in FLIPLR_TEMPLATES
]

ROT180_TEMPLATES = [
    (
        np.array(
            [
                [0, 0, 0, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 0, 0, 0],
            ],
            dtype=bool,
        ),
        (3, 0),
    ),
    (
        np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        (3, 0),
    ),
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _tight_crop_grid(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(grid: np.ndarray, bg: int = BG) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    H, W = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _symmetry_types(mask: np.ndarray) -> list[str]:
    types = []
    if np.array_equal(mask, np.fliplr(mask)):
        types.append(TYPE_FLIPLR)
    if np.array_equal(mask, np.flipud(mask)):
        types.append(TYPE_FLIPUD)
    if np.array_equal(mask, np.rot90(mask, 2)):
        types.append(TYPE_ROT180)
    return types


def _boundary_count(mask: np.ndarray, side: str) -> int:
    if side == "left":
        return int(mask[:, 0].sum())
    if side == "right":
        return int(mask[:, -1].sum())
    if side == "top":
        return int(mask[0, :].sum())
    if side == "bottom":
        return int(mask[-1, :].sum())
    raise ValueError(side)


def _inward_neighbor(side: str, anchor: tuple[int, int]) -> tuple[int, int]:
    r, c = anchor
    if side == "left":
        return r, c + 1
    if side == "right":
        return r, c - 1
    if side == "top":
        return r + 1, c
    if side == "bottom":
        return r - 1, c
    raise ValueError(side)


def _random_anchor(h: int, w: int, side: str) -> tuple[int, int]:
    if side in {"left", "right"}:
        r = random.randint(1, h - 2)
        c = 0 if side == "left" else w - 1
        return r, c
    r = 0 if side == "top" else h - 1
    c = random.randint(1, w - 2)
    return r, c


def _forbidden_side_mask(h: int, w: int, side: str, anchor: tuple[int, int]) -> np.ndarray:
    allowed = np.ones((h, w), dtype=bool)
    if side == "left":
        allowed[:, 0] = False
    elif side == "right":
        allowed[:, -1] = False
    elif side == "top":
        allowed[0, :] = False
    elif side == "bottom":
        allowed[-1, :] = False
    else:
        raise ValueError(side)
    allowed[anchor] = True
    return allowed


def _make_patch(mask: np.ndarray, connector: tuple[int, int], color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    patch[connector] = int(CONNECTOR_COLOR)
    return patch


def _translate_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _component_info_from_grid(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    rel_coords = coords - np.array([r0, c0], dtype=int)
    mask = patch != BG
    connector_coords = np.argwhere(patch == CONNECTOR_COLOR)
    if connector_coords.shape[0] != 1:
        raise AssertionError("each object must have exactly one connector")
    body_colors = sorted({int(v) for v in patch[(patch != BG) & (patch != CONNECTOR_COLOR)]})
    if len(body_colors) != 1:
        raise AssertionError("each object must be monochromatic apart from connector")
    return {
        "coords": coords.copy(),
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "connector_local": tuple(map(int, connector_coords[0])),
        "connector_global": tuple(map(int, connector_coords[0] + np.array([r0, c0], dtype=int))),
        "body_color": body_colors[0],
        "body_area": int(np.sum((patch != BG) & (patch != CONNECTOR_COLOR))),
        "area": int(mask.sum()),
        "symmetry_types": _symmetry_types(mask),
    }


def _merge_base_and_plug(base: dict, plug: dict, symmetry_type: str) -> np.ndarray:
    dr, dc = TYPE_TO_DELTA[symmetry_type]
    br, bc = base["connector_global"]
    pr, pc = plug["connector_global"]
    shift_r = int(br + dr - pr)
    shift_c = int(bc + dc - pc)

    color_map: dict[tuple[int, int], int] = {}
    for r, c in base["coords"]:
        color_map[(int(r), int(c))] = int(base["patch"][int(r - base["bbox"][0]), int(c - base["bbox"][1])])
    for r, c in plug["coords"]:
        rr = int(r + shift_r)
        cc = int(c + shift_c)
        color_map[(rr, cc)] = int(plug["patch"][int(r - plug["bbox"][0]), int(c - plug["bbox"][1])])

    for key, value in list(color_map.items()):
        if int(value) == int(CONNECTOR_COLOR):
            color_map[key] = int(plug["body_color"])

    coords = np.array(list(color_map.keys()), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in color_map.items():
        out[int(r - r0), int(c - c0)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the uniquely symmetric plug sprite, dock it in the direction encoded by its symmetry, and tight-crop the merge.

    Input structure:
    - The grid has black background, one largest blue base sprite, and 3-4 smaller candidate sprites.
    - Every sprite is a 4-connected monochromatic body with exactly one red connector pixel on its edge.
    - Candidate sprites are separated from one another and from the base.

    Rule:
    - Detect all non-black 4-connected components and identify the base as the unique largest body.
    - Crop each candidate to its tight bounding box and test the occupied shape for exactly one of: left-right reflection, up-down reflection, or 180-degree rotation.
    - Keep the unique symmetric candidate, translate it so its connector lands one cell right / down / up of the base connector according to that symmetry type, then discard the other candidates.
    - Recolor both connector pixels to the plug's body color and return the tight crop of the merged base-plus-plug object.

    Key insight:
    - The same symmetry property both selects which candidate matters and encodes the docking direction, so shape analysis is needed twice but with the same evidence.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    infos = [_component_info_from_grid(g, coords) for coords in _component_coords(g, bg=BG)]
    if len(infos) < 4:
        raise AssertionError("expected one base and multiple candidates")

    max_body = max(info["body_area"] for info in infos)
    bases = [info for info in infos if info["body_area"] == max_body]
    if len(bases) != 1:
        raise AssertionError("expected a unique largest base")
    base = bases[0]

    candidates = [info for info in infos if info is not base]
    selected = [info for info in candidates if len(info["symmetry_types"]) == 1]
    if len(selected) != 1:
        raise AssertionError("expected exactly one symmetric candidate")
    plug = selected[0]
    symmetry_type = plug["symmetry_types"][0]

    return _tight_crop_grid(_merge_base_and_plug(base, plug, symmetry_type), bg=BG)


def _generate_random_asymmetric_patch(color: int, side: str, min_area: int, max_area: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        anchor = _random_anchor(h, w, side)
        inward = _inward_neighbor(side, anchor)
        allowed = _forbidden_side_mask(h, w, side, anchor)
        mask = np.zeros((h, w), dtype=bool)
        mask[anchor] = True
        mask[inward] = True
        target = random.randint(min_area, min(max_area, h * w - max(0, _boundary_count(~allowed, side) - 1)))
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != target or not _is_connected(mask):
            continue
        cropped, (r0, c0) = _tight_crop_mask(mask)
        conn = (anchor[0] - r0, anchor[1] - c0)
        if _boundary_count(cropped, side) != 1:
            continue
        if len(_symmetry_types(cropped)) != 0:
            continue
        if int(cropped.sum()) < min_area or int(cropped.sum()) > max_area:
            continue
        return _make_patch(cropped, conn, color)
    return None


def _generate_symmetric_patch(color: int, symmetry_type: str) -> np.ndarray:
    if symmetry_type == TYPE_FLIPLR:
        mask, conn = random.choice(FLIPLR_TEMPLATES)
    elif symmetry_type == TYPE_FLIPUD:
        mask, conn = random.choice(FLIPUD_TEMPLATES)
    elif symmetry_type == TYPE_ROT180:
        mask, conn = random.choice(ROT180_TEMPLATES)
    else:
        raise ValueError(symmetry_type)
    return _make_patch(mask.copy(), conn, color)


def _generate_base_patch(symmetry_type: str) -> np.ndarray | None:
    base_side, _plug_side = TYPE_TO_SIDES[symmetry_type]
    return _generate_random_asymmetric_patch(BASE_COLOR, base_side, min_area=10, max_area=15)


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 2) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if np.any((canvas[r0 : r0 + h, c0 : c0 + w] != BG) & (patch != BG)):
        return False
    r1 = max(0, r0 - padding)
    c1 = max(0, c0 - padding)
    r2 = min(H, r0 + h + padding)
    c2 = min(W, c0 + w + padding)
    return not np.any((canvas[r1:r2, c1:c2] != BG) & (np.pad(patch != BG, ((r0 - r1, r2 - (r0 + h)), (c0 - c1, c2 - (c0 + w))), constant_values=False)))


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    view[patch != BG] = patch[patch != BG]


def _random_location(canvas: np.ndarray, patch: np.ndarray, padding: int = 2) -> tuple[int, int] | None:
    H, W = canvas.shape
    h, w = patch.shape
    spots = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(spots)
    for r, c in spots:
        if _can_place(canvas, patch, r, c, padding=padding):
            return r, c
    return None


def _expected_from_metadata(base_patch: np.ndarray, base_rc: tuple[int, int], plug_patch: np.ndarray, plug_rc: tuple[int, int], symmetry_type: str) -> np.ndarray:
    base_coords = np.argwhere(base_patch != BG) + np.array(base_rc, dtype=int)
    plug_coords = np.argwhere(plug_patch != BG) + np.array(plug_rc, dtype=int)
    base_info = {
        "coords": base_coords,
        "bbox": (base_rc[0], base_rc[1], base_rc[0] + base_patch.shape[0] - 1, base_rc[1] + base_patch.shape[1] - 1),
        "patch": base_patch,
        "connector_global": tuple(map(int, np.argwhere(base_patch == CONNECTOR_COLOR)[0] + np.array(base_rc, dtype=int))),
        "body_color": BASE_COLOR,
    }
    plug_color = int(next(v for v in sorted(np.unique(plug_patch)) if v not in {BG, CONNECTOR_COLOR}))
    plug_info = {
        "coords": plug_coords,
        "bbox": (plug_rc[0], plug_rc[1], plug_rc[0] + plug_patch.shape[0] - 1, plug_rc[1] + plug_patch.shape[1] - 1),
        "patch": plug_patch,
        "connector_global": tuple(map(int, np.argwhere(plug_patch == CONNECTOR_COLOR)[0] + np.array(plug_rc, dtype=int))),
        "body_color": plug_color,
    }
    return _merge_base_and_plug(base_info, plug_info, symmetry_type)


def _validate_generated_scene(grid: np.ndarray, symmetry_type: str) -> None:
    infos = [_component_info_from_grid(grid, coords) for coords in _component_coords(grid, bg=BG)]
    assert 4 <= len(infos) <= 5
    body_areas = sorted(info["body_area"] for info in infos)
    assert body_areas[-1] > body_areas[-2]

    max_body = max(info["body_area"] for info in infos)
    base = [info for info in infos if info["body_area"] == max_body][0]
    candidates = [info for info in infos if info is not base]
    selected = [info for info in candidates if info["symmetry_types"] == [symmetry_type]]
    assert len(selected) == 1
    for info in candidates:
        if info is not selected[0]:
            assert info["symmetry_types"] == []

    base_side, plug_side = TYPE_TO_SIDES[symmetry_type]
    base_conn = selected[0]["connector_local"]
    assert _boundary_count(selected[0]["mask"], plug_side) == 1
    assert _boundary_count(base["mask"], base_side) == 1
    assert selected[0]["mask"][base_conn]


def _generate_attempt() -> np.ndarray | None:
    symmetry_type = random.choice([TYPE_FLIPLR, TYPE_FLIPUD, TYPE_ROT180])
    base_patch = _generate_base_patch(symmetry_type)
    if base_patch is None:
        return None

    n_candidates = random.randint(3, 4)
    colors = random.sample(CANDIDATE_COLORS, n_candidates)
    chosen_index = random.randrange(n_candidates)
    candidate_patches = []
    for idx in range(n_candidates):
        if idx == chosen_index:
            candidate_patches.append(_generate_symmetric_patch(colors[idx], symmetry_type))
        else:
            patch = _generate_random_asymmetric_patch(colors[idx], random.choice(["left", "right", "top", "bottom"]), 4, 8)
            if patch is None:
                return None
            candidate_patches.append(patch)

    H = random.randint(18, 24)
    W = random.randint(18, 24)
    canvas = np.zeros((H, W), dtype=int)

    base_rc = _random_location(canvas, base_patch, padding=2)
    if base_rc is None:
        return None
    _paste(canvas, base_patch, *base_rc)

    candidate_rcs: list[tuple[int, int]] = []
    shuffled = list(range(n_candidates))
    random.shuffle(shuffled)
    for idx in shuffled:
        rc = _random_location(canvas, candidate_patches[idx], padding=2)
        if rc is None:
            return None
        _paste(canvas, candidate_patches[idx], *rc)
        candidate_rcs.append((idx, rc))

    placed_rcs = {idx: rc for idx, rc in candidate_rcs}
    expected = _expected_from_metadata(
        base_patch,
        base_rc,
        candidate_patches[chosen_index],
        placed_rcs[chosen_index],
        symmetry_type,
    )
    if max(expected.shape, default=0) > 30:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        _validate_generated_scene(canvas, symmetry_type)
    except AssertionError:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
