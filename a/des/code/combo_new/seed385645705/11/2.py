from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
FRAME_COUNT_RANGE = (2, 4)
GRID_SIDE_RANGE = (22, 30)
MARGIN_RANGE = (1, 3)
HOLE_AREA_RANGE = (4, 8)
HOLE_CANVAS_RANGE = (2, 5)
PLACEMENT_PADDING = 1
PLACEMENT_MARGIN = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _count_holes(mask: np.ndarray) -> int:
    inv = ~mask
    holes = 0
    for comp in _components(inv):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 1] == 0):
            continue
        if np.any(comp[:, 0] == mask.shape[0] - 1) or np.any(comp[:, 1] == mask.shape[1] - 1):
            continue
        holes += 1
    return holes


def _rotation_keys(mask: np.ndarray) -> list[tuple[tuple[int, int], bytes]]:
    keys: list[tuple[tuple[int, int], bytes]] = []
    for k in range(4):
        rotated = _crop_mask(np.rot90(mask, k))
        keys.append((rotated.shape, rotated.tobytes()))
    return keys


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _extract_frame_hole(
    grid: np.ndarray, color: int, coords: np.ndarray
) -> tuple[tuple[int, int], np.ndarray] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != BG) & (sub != int(color))):
        return None
    if not np.all(sub[0, :] == int(color)):
        return None
    if not np.all(sub[-1, :] == int(color)):
        return None
    if not np.all(sub[:, 0] == int(color)):
        return None
    if not np.all(sub[:, -1] == int(color)):
        return None

    holes: list[np.ndarray] = []
    for hole_coords in _components(sub == BG):
        if np.any(hole_coords[:, 0] == 0) or np.any(hole_coords[:, 1] == 0):
            continue
        if np.any(hole_coords[:, 0] == h - 1) or np.any(hole_coords[:, 1] == w - 1):
            continue
        holes.append(hole_coords)
    if len(holes) != 1:
        return None

    hr0, hc0, hr1, hc1 = _bbox(holes[0])
    hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
    rel = holes[0] - np.array([hr0, hc0])
    hole_mask[rel[:, 0], rel[:, 1]] = True
    return (int(hr0), int(hc0)), hole_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each outside fragment back into the matching same-color frame hole.

    Input structure:
    - The background is black.
    - Several monochrome template objects have a rectangular outer border and one enclosed black hole inside.
    - Outside the templates, each template color also appears as one separate monochrome fragment on the background.
    - Each fragment has the same area as its matching hole and may be a rotated copy of that hole shape.

    Rule:
    - Pair every outside fragment with the template of the same color.
    - Extract the template's enclosed black hole, rotate the fragment by 0, 90, 180, or 270 degrees until its mask matches the hole, then paint that hole with the template color.
    - Erase the fragment from its original location and keep the surrounding rectangular border unchanged.

    Key insight:
    Color chooses the destination template, while shape matching across the four rotations determines how the fragment fits the hole.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames: dict[int, dict[str, object]] = {}
    fragments: dict[int, dict[str, object]] = {}

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        comps = _components(g == int(color))
        frame_candidates: list[dict[str, object]] = []
        fragment_candidates: list[dict[str, object]] = []
        for coords in comps:
            comp_bbox, comp_mask = _coords_to_mask(coords)
            hole_info = _extract_frame_hole(g, color, coords)
            payload = {
                "coords": coords,
                "bbox": comp_bbox,
                "mask": comp_mask,
            }
            if hole_info is None:
                fragment_candidates.append(payload)
            else:
                hole_origin, hole_mask = hole_info
                payload["hole_origin"] = hole_origin
                payload["hole_mask"] = hole_mask
                frame_candidates.append(payload)

        assert len(frame_candidates) == 1, f"expected one frame for color {color}"
        assert len(fragment_candidates) == 1, f"expected one fragment for color {color}"
        frames[color] = frame_candidates[0]
        fragments[color] = fragment_candidates[0]

    for color in sorted(frames):
        frame = frames[color]
        fragment = fragments[color]
        hole_mask = frame["hole_mask"]
        frag_mask = fragment["mask"]
        assert isinstance(hole_mask, np.ndarray)
        assert isinstance(frag_mask, np.ndarray)

        matches = [
            k
            for k in range(4)
            if np.rot90(frag_mask, k).shape == hole_mask.shape
            and np.array_equal(np.rot90(frag_mask, k), hole_mask)
        ]
        assert matches, f"no rotation fits color {color}"

        frag_coords = fragment["coords"]
        assert isinstance(frag_coords, np.ndarray)
        out[frag_coords[:, 0], frag_coords[:, 1]] = BG

        frame_r0, frame_c0, _frame_r1, _frame_c1 = frame["bbox"]
        hole_r0, hole_c0 = frame["hole_origin"]
        target = out[
            frame_r0 + hole_r0 : frame_r0 + hole_r0 + hole_mask.shape[0],
            frame_c0 + hole_c0 : frame_c0 + hole_c0 + hole_mask.shape[1],
        ]
        target[hole_mask] = int(color)

    return out


FALLBACK_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
]


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        canvas_h = random.randint(*HOLE_CANVAS_RANGE)
        canvas_w = random.randint(*HOLE_CANVAS_RANGE)
        max_area = min(HOLE_AREA_RANGE[1], canvas_h * canvas_w - 1)
        if max_area < HOLE_AREA_RANGE[0]:
            continue
        target_area = random.randint(HOLE_AREA_RANGE[0], max_area)

        start = (random.randrange(canvas_h), random.randrange(canvas_w))
        cells = {start}
        frontier = set()
        for dr, dc in DIR4:
            rr = start[0] + dr
            cc = start[1] + dc
            if 0 <= rr < canvas_h and 0 <= cc < canvas_w:
                frontier.add((rr, cc))

        while len(cells) < target_area and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            cells.add((rr, cc))
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target_area:
            continue

        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        mask = _crop_mask(mask)

        if min(mask.shape) < 2:
            continue
        if _count_holes(mask) != 0:
            continue
        if len(set(_rotation_keys(mask))) != 4:
            continue
        return mask

    return random.choice(FALLBACK_SHAPES).copy()


def _build_spec(color: int) -> dict[str, object]:
    hole_mask = _random_shape()
    hole_h, hole_w = hole_mask.shape
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    frame = np.full((hole_h + top + bottom, hole_w + left + right), int(color), dtype=int)
    frame[top : top + hole_h, left : left + hole_w][hole_mask] = BG

    rotation = random.randrange(4)
    fragment_mask = np.rot90(hole_mask, rotation)
    fragment = _mask_to_patch(fragment_mask, color)

    return {
        "color": int(color),
        "frame": frame,
        "fragment": fragment,
        "hole_mask": hole_mask,
        "hole_origin": (int(top), int(left)),
        "rotation": int(rotation),
    }


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(PLACEMENT_MARGIN, h - ph - PLACEMENT_MARGIN + 1):
        for c0 in range(PLACEMENT_MARGIN, w - pw - PLACEMENT_MARGIN + 1):
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(h, r0 + ph + PLACEMENT_PADDING)
            cc1 = min(w, c0 + pw + PLACEMENT_PADDING)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(h, r0 + ph + PLACEMENT_PADDING)
    cc1 = min(w, c0 + pw + PLACEMENT_PADDING)
    reserved[rr0:rr1, cc0:cc1] = True
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    return int(r0), int(c0)


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        color = int(item["color"])
        frag_r0, frag_c0 = item["fragment_pos"]
        fragment = item["fragment"]
        frag_mask = fragment != BG
        out[frag_r0 : frag_r0 + fragment.shape[0], frag_c0 : frag_c0 + fragment.shape[1]][frag_mask] = BG

        frame_r0, frame_c0 = item["frame_pos"]
        hole_r0, hole_c0 = item["hole_origin"]
        hole_mask = item["hole_mask"]
        target = out[
            frame_r0 + hole_r0 : frame_r0 + hole_r0 + hole_mask.shape[0],
            frame_c0 + hole_c0 : frame_c0 + hole_c0 + hole_mask.shape[1],
        ]
        target[hole_mask] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        frame_count = random.randint(*FRAME_COUNT_RANGE)
        colors = random.sample(PALETTE, frame_count)
        specs = [_build_spec(color) for color in colors]

        max_h = max(
            max(spec["frame"].shape[0], spec["fragment"].shape[0]) for spec in specs
        )
        max_w = max(
            max(spec["frame"].shape[1], spec["fragment"].shape[1]) for spec in specs
        )
        min_side = max(
            GRID_SIDE_RANGE[0],
            max_h + 2 * (PLACEMENT_MARGIN + PLACEMENT_PADDING),
            max_w + 2 * (PLACEMENT_MARGIN + PLACEMENT_PADDING),
        )
        if min_side > GRID_SIDE_RANGE[1]:
            continue

        grid_h = random.randint(min_side, GRID_SIDE_RANGE[1])
        grid_w = random.randint(min_side, GRID_SIDE_RANGE[1])
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        reserved = np.zeros((grid_h, grid_w), dtype=bool)

        placements: list[dict[str, object]] = []
        order = sorted(
            specs,
            key=lambda spec: spec["frame"].shape[0] * spec["frame"].shape[1],
            reverse=True,
        )
        ok = True
        for spec in order:
            frame_pos = _place_patch(grid, reserved, spec["frame"])
            if frame_pos is None:
                ok = False
                break
            fragment_pos = _place_patch(grid, reserved, spec["fragment"])
            if fragment_pos is None:
                ok = False
                break

            placed = dict(spec)
            placed["frame_pos"] = frame_pos
            placed["fragment_pos"] = fragment_pos
            placements.append(placed)

        if not ok:
            continue

        expected = _expected_output(grid, placements)
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
