from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
TEAL = 8
OTHER_COLORS = (1, 2, 3, 4, 6, 7, 9)
RETRY_LIMIT = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "mask": _crop_mask_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _shifted_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[rr + top, cc + left] = int(color)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for base in (mask, np.fliplr(mask), np.flipud(mask)):
        for k in range(4):
            variants.append(_normalize_mask(np.rot90(base, k)))
    unique: list[np.ndarray] = []
    for variant in variants:
        if not any(np.array_equal(variant, seen) for seen in unique):
            unique.append(variant)
    return unique


def _is_irregular(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    if int(mask.sum()) == h * w:
        return False
    return True


def _is_asymmetric(mask: np.ndarray) -> bool:
    transforms = _mask_transforms(mask)
    return len(transforms) > 1


def _random_connected_mask() -> np.ndarray:
    area = random.randint(3, 6)
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (-1, 0), (0, -1)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(*cell):
            if nb not in cells:
                frontier.add(nb)
    coords = np.array(sorted(cells), dtype=int)
    coords -= coords.min(axis=0)
    mask = np.zeros((coords[:, 0].max() + 1, coords[:, 1].max() + 1), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return _normalize_mask(mask)


def _generate_prototype_mask() -> np.ndarray:
    for _ in range(200):
        mask = _random_connected_mask()
        if _is_irregular(mask) and _is_asymmetric(mask):
            return mask
    raise RuntimeError("failed to generate prototype mask")


def _random_nonmatching_mask(proto: np.ndarray, *, prefer_same_area: bool = False, prefer_same_bbox: bool = False) -> np.ndarray:
    proto_area = int(proto.sum())
    proto_shape = proto.shape
    for _ in range(300):
        mask = _random_connected_mask()
        if np.array_equal(mask, proto):
            continue
        if prefer_same_area and int(mask.sum()) != proto_area:
            continue
        if prefer_same_bbox and mask.shape != proto_shape:
            continue
        return mask
    raise RuntimeError("failed to generate nonmatching mask")


def _touch_kind_edges(kind: str) -> tuple[str, ...]:
    mapping = {
        "top": ("top",),
        "bottom": ("bottom",),
        "left": ("left",),
        "right": ("right",),
        "tl": ("top", "left"),
        "tr": ("top", "right"),
        "bl": ("bottom", "left"),
        "br": ("bottom", "right"),
    }
    return mapping[kind]


def _sample_bbox_for_kind(H: int, W: int, mask: np.ndarray, kind: str) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    if h >= H - 1 or w >= W - 1:
        return None
    if kind == "interior":
        if H - h - 2 < 1 or W - w - 2 < 1:
            return None
        r0 = random.randint(1, H - h - 2)
        c0 = random.randint(1, W - w - 2)
    elif kind == "top":
        if W - w - 2 < 1:
            return None
        r0 = 0
        c0 = random.randint(1, W - w - 2)
    elif kind == "bottom":
        if W - w - 2 < 1:
            return None
        r0 = H - h
        c0 = random.randint(1, W - w - 2)
    elif kind == "left":
        if H - h - 2 < 1:
            return None
        r0 = random.randint(1, H - h - 2)
        c0 = 0
    elif kind == "right":
        if H - h - 2 < 1:
            return None
        r0 = random.randint(1, H - h - 2)
        c0 = W - w
    elif kind == "tl":
        r0, c0 = 0, 0
    elif kind == "tr":
        r0, c0 = 0, W - w
    elif kind == "bl":
        r0, c0 = H - h, 0
    elif kind == "br":
        r0, c0 = H - h, W - w
    else:
        raise ValueError(f"unknown placement kind: {kind}")
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _shift_for_bbox(H: int, W: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h, w = _bbox_size(bbox)
    dr = (h if r0 == 0 else 0) + (-h if r1 == H - 1 else 0)
    dc = (w if c0 == 0 else 0) + (-w if c1 == W - 1 else 0)
    return dr, dc


def _classify_components(grid: np.ndarray) -> tuple[dict, list[dict]]:
    comps = _find_components(grid)
    prototype_candidates = [comp for comp in comps if comp["color"] == TEAL]
    assert len(prototype_candidates) == 1, "expected exactly one teal prototype"
    prototype = prototype_candidates[0]
    pr0, pc0, pr1, pc1 = prototype["bbox"]
    H, W = grid.shape
    assert 0 < pr0 and 0 < pc0 and pr1 < H - 1 and pc1 < W - 1, "prototype must be internal"

    labeled: list[dict] = []
    for comp in comps:
        if comp is prototype:
            continue
        bbox = comp["bbox"]
        h, w = _bbox_size(bbox)
        match = bool(comp["mask"].shape == prototype["mask"].shape and np.array_equal(comp["mask"], prototype["mask"]))
        dr, dc = _shift_for_bbox(H, W, bbox)
        labeled.append(
            {
                **comp,
                "match": match,
                "touches_boundary": bool(dr or dc),
                "touch_count": int((bbox[0] == 0) + (bbox[1] == 0) + (bbox[2] == H - 1) + (bbox[3] == W - 1)),
                "shift": (dr, dc),
                "size": (h, w),
            }
        )
    return prototype, labeled


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    prototype, others = _classify_components(g)
    out = g.copy()

    for comp in others:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for comp in others:
        if not comp["match"]:
            continue
        coords = comp["coords"]
        dr, dc = comp["shift"]
        if dr or dc:
            shifted = coords + np.array([dr, dc], dtype=int)
            assert np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < H) & (0 <= shifted[:, 1]) & (shifted[:, 1] < W))
            out[shifted[:, 0], shifted[:, 1]] = TEAL
        else:
            out[coords[:, 0], coords[:, 1]] = TEAL

    pr_coords = prototype["coords"]
    out[pr_coords[:, 0], pr_coords[:, 1]] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only prototype-matching sprites, recolor them teal, and push only the boundary-touching matches inward.

    Input structure:
    - A black background contains one internal teal prototype sprite and several other monochrome 4-connected sprites.
    - Every non-prototype sprite has a unique color, while shape comparison ignores color and uses the cropped foreground mask.
    - Some non-prototype sprites touch one grid edge or a corner, and others lie fully inside the grid.

    Rule:
    - Use the teal prototype's cropped binary mask as the exact template, with no rotation or reflection allowed.
    - Erase every non-prototype sprite whose mask does not exactly match that template.
    - For each matching sprite, recolor it teal; if it touches the border, translate it inward by one bounding-box height vertically and/or one bounding-box width horizontally according to the touched edge or edges, then erase its original location.
    - Keep the teal prototype unchanged, and let moved teal pixels overwrite anything already kept at their destination.

    Key insight:
    - Shape matching decides survival, but border contact decides whether a surviving copy stays put or is pushed inward, so both conditions are necessary.
    """
    return _solve(np.asarray(grid, dtype=int))


def _build_expected_from_specs(H: int, W: int, placed: list[dict]) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for spec in placed:
        if spec["role"] == "prototype":
            _paint_mask(out, spec["mask"], spec["bbox"][0], spec["bbox"][1], TEAL)
        elif spec["match"]:
            bbox = spec["bbox"]
            dr, dc = _shift_for_bbox(H, W, bbox)
            target_bbox = _shifted_bbox(bbox, dr, dc)
            _paint_mask(out, spec["mask"], target_bbox[0], target_bbox[1], TEAL)
    return out


def _pick_extra_role() -> tuple[bool, str]:
    options = [
        (True, random.choice(("interior", "top", "bottom", "left", "right"))),
        (True, random.choice(("tl", "tr", "bl", "br"))),
        (False, "interior"),
        (False, random.choice(("top", "bottom", "left", "right", "tl", "tr", "bl", "br"))),
    ]
    return random.choice(options)


def _place_specs(H: int, W: int, specs: list[dict]) -> list[dict] | None:
    ordered = sorted(
        specs,
        key=lambda spec: (
            spec["role"] != "prototype",
            spec["kind"] == "interior",
            random.random(),
        ),
    )
    placed: list[dict] = []
    input_bboxes: list[tuple[int, int, int, int]] = []
    kept_output_bboxes: list[tuple[int, int, int, int]] = []

    for spec in ordered:
        mask = spec["mask"]
        for _ in range(250):
            bbox = _sample_bbox_for_kind(H, W, mask, spec["kind"])
            if bbox is None:
                return None
            if any(not _bbox_separated(bbox, old, pad=1) for old in input_bboxes):
                continue

            output_bbox = None
            if spec["role"] == "prototype":
                output_bbox = bbox
            elif spec["match"]:
                dr, dc = _shift_for_bbox(H, W, bbox)
                output_bbox = _shifted_bbox(bbox, dr, dc)
                if any(not _bbox_separated(output_bbox, old, pad=1) for old in kept_output_bboxes):
                    continue
            placed_spec = dict(spec)
            placed_spec["bbox"] = bbox
            placed.append(placed_spec)
            input_bboxes.append(bbox)
            if output_bbox is not None:
                kept_output_bboxes.append(output_bbox)
            break
        else:
            return None
    return placed


def _validate_semantics(inp: np.ndarray, placed: list[dict]) -> bool:
    H, W = inp.shape
    prototype, others = _classify_components(inp)
    if not (4 <= len(others) <= 6):
        return False

    match_boundary = [comp for comp in others if comp["match"] and comp["touches_boundary"]]
    match_interior = [comp for comp in others if comp["match"] and not comp["touches_boundary"]]
    nonmatch = [comp for comp in others if not comp["match"]]
    if not match_boundary or not match_interior or not nonmatch:
        return False
    if not any(comp["touch_count"] == 2 for comp in match_boundary):
        return False
    if not any(comp["touch_count"] == 1 for comp in match_boundary):
        return False
    if not any(comp["touches_boundary"] for comp in nonmatch):
        return False

    proto_mask = prototype["mask"]
    transformed_decoys = 0
    proto_area = int(proto_mask.sum())
    proto_shape = proto_mask.shape
    for comp in nonmatch:
        if int(comp["mask"].sum()) == proto_area or comp["mask"].shape == proto_shape:
            transformed_decoys += 1
        if any(np.array_equal(comp["mask"], t) for t in _mask_transforms(proto_mask) if not np.array_equal(t, proto_mask)):
            transformed_decoys += 1
            break
    if transformed_decoys == 0:
        return False

    expected = _build_expected_from_specs(H, W, placed)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, inp):
        return False

    for comp in others:
        coords = comp["coords"]
        vals = actual[coords[:, 0], coords[:, 1]]
        if comp["match"] and not comp["touches_boundary"]:
            if not np.all(vals == TEAL):
                return False
        elif not np.all(vals == BLACK):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    total_others = random.randint(4, 6)
    H = random.randint(13, 20)
    W = random.randint(13, 20)

    prototype_mask = _generate_prototype_mask()
    transformed_variants = [m for m in _mask_transforms(prototype_mask) if not np.array_equal(m, prototype_mask)]
    if not transformed_variants:
        return None
    decoy_mask = random.choice(transformed_variants)

    specs: list[dict] = [
        {"role": "prototype", "match": True, "kind": "interior", "mask": prototype_mask, "color": TEAL},
        {"role": "match_corner", "match": True, "kind": random.choice(("tl", "tr", "bl", "br")), "mask": prototype_mask},
        {"role": "match_edge", "match": True, "kind": random.choice(("top", "bottom", "left", "right")), "mask": prototype_mask},
        {"role": "match_interior", "match": True, "kind": "interior", "mask": prototype_mask},
        {
            "role": "nonmatch_boundary",
            "match": False,
            "kind": random.choice(("top", "bottom", "left", "right", "tl", "tr", "bl", "br")),
            "mask": decoy_mask,
        },
    ]

    need_same_area_decoy = True
    need_same_bbox_decoy = True
    while len(specs) - 1 < total_others:
        match, kind = _pick_extra_role()
        if match:
            mask = prototype_mask
        else:
            mask = _random_nonmatching_mask(
                prototype_mask,
                prefer_same_area=need_same_area_decoy,
                prefer_same_bbox=need_same_bbox_decoy and random.random() < 0.7,
            )
            need_same_area_decoy = False
            if mask.shape == prototype_mask.shape:
                need_same_bbox_decoy = False
        specs.append({"role": "extra", "match": match, "kind": kind, "mask": mask})

    colors = random.sample(OTHER_COLORS, total_others)
    color_index = 0
    for spec in specs:
        if spec["role"] == "prototype":
            continue
        spec["color"] = colors[color_index]
        color_index += 1

    placed = _place_specs(H, W, specs)
    if placed is None:
        return None

    inp = np.zeros((H, W), dtype=int)
    for spec in placed:
        _paint_mask(inp, spec["mask"], spec["bbox"][0], spec["bbox"][1], spec["color"])

    if not _validate_semantics(inp, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective inward-push ARC input")
