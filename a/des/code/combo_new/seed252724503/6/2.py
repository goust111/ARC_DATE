import random

import numpy as np


BLACK = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
AREA_TO_DIMS = {
    3: ((2, 2),),
    4: ((2, 3), (3, 2)),
    5: ((2, 4), (3, 3), (4, 2)),
}
GEN_RETRIES = 400

_GEN_CALL_COUNT = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _canonical_l(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _is_l_shape(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    h, w = mask.shape
    if not (3 <= area <= 5):
        return False
    if h < 2 or w < 2:
        return False
    if area != h + w - 1:
        return False
    base = _canonical_l(h, w)
    for k in range(4):
        if np.array_equal(mask, np.rot90(base, k)):
            return True
    return False


def _classify_component(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    mask, (r0, c0, r1, c1) = _tight_mask(coords)
    h, w = mask.shape
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    if h >= 4 and w >= 4:
        if (
            mask[0, :].all()
            and mask[-1, :].all()
            and mask[:, 0].all()
            and mask[:, -1].all()
            and np.all((sub == BLACK) | (sub == color))
        ):
            hole = ~mask[1:-1, 1:-1]
            hole_coords = _components(hole)
            if len(hole_coords) == 1:
                cavity_mask, (hr0, hc0, hr1, hc1) = _tight_mask(hole_coords[0])
                if _is_l_shape(cavity_mask):
                    cavity_coords = [
                        (int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in hole_coords[0]
                    ]
                    return {
                        "kind": "container",
                        "color": int(color),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "cavity_mask": cavity_mask,
                        "cavity_bbox": (int(r0 + 1 + hr0), int(c0 + 1 + hc0), int(r0 + 1 + hr1), int(c0 + 1 + hc1)),
                        "cavity_coords": cavity_coords,
                    }

    if _is_l_shape(mask):
        return {
            "kind": "piece",
            "color": int(color),
            "coords": coords,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "mask": mask,
        }
    return None


def _parse_scene(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    containers: dict[int, dict[str, object]] = {}
    pieces: dict[int, dict[str, object]] = {}
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BLACK)
    for color in colors:
        for coords in _components(grid == color):
            obj = _classify_component(grid, int(color), coords)
            assert obj is not None, f"unexpected component for color {color}"
            if obj["kind"] == "container":
                assert color not in containers, f"duplicate container color {color}"
                containers[color] = obj
            else:
                assert color not in pieces, f"duplicate piece color {color}"
                pieces[color] = obj
    assert containers, "expected at least one container"
    assert set(containers) == set(pieces), "each container color must have one matching piece"
    return containers, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered L piece into the same-colored rectangular container cavity.

    Input structure:
    - Background is black.
    - Several monochrome rectangular containers have one-pixel borders and same-colored partial fill.
    - Each container contains exactly one black 4-connected L-shaped cavity of 3 to 5 cells.
    - Outside the containers, each container color appears once more as a separate monochrome L piece.

    Rule:
    - Use color to pair every outside L piece with its matching container.
    - For each pair, try the four quarter-turn rotations and find the unique rotation whose shape matches the container's black cavity exactly.
    - Erase the piece from its original location back to black and paint that color into the matched cavity.
    - Keep the container border and all pre-existing filled cells unchanged.

    Key insight:
    - Color tells which piece belongs to which container, but the actual placement is determined by recognizing that each black cavity is the rotated piece's exact mold.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers, pieces = _parse_scene(g)
    out = g.copy()

    for color in sorted(containers):
        container = containers[color]
        piece = pieces[color]
        matches = [
            k
            for k in range(4)
            if np.array_equal(np.rot90(piece["mask"], k), container["cavity_mask"])
        ]
        assert len(matches) == 1, f"color {color} must have exactly one valid rotation"

        for r, c in piece["coords"]:
            out[int(r), int(c)] = BLACK
        for r, c in container["cavity_coords"]:
            out[int(r), int(c)] = int(color)

    return out


def _sample_l_mask(area: int, rotation: int) -> np.ndarray:
    height, width = random.choice(AREA_TO_DIMS[area])
    return np.rot90(_canonical_l(height, width), rotation)


def _build_pair_spec(color: int, forced_rotation: int | None) -> dict[str, object]:
    area = random.choice((3, 4, 5))
    cavity_rotation = random.randrange(4)
    required_rotation = forced_rotation if forced_rotation is not None else random.randrange(4)
    cavity_mask = _sample_l_mask(area, cavity_rotation)
    piece_mask = np.rot90(cavity_mask, -required_rotation)

    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)

    cavity_h, cavity_w = cavity_mask.shape
    inner_h = cavity_h + top_pad + bottom_pad
    inner_w = cavity_w + left_pad + right_pad

    container_patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    container_patch[0, :] = color
    container_patch[-1, :] = color
    container_patch[:, 0] = color
    container_patch[:, -1] = color
    container_patch[1:-1, 1:-1] = color

    cavity_r0 = 1 + top_pad
    cavity_c0 = 1 + left_pad
    cavity_cells = []
    for rr, cc in np.argwhere(cavity_mask):
        rr = int(rr)
        cc = int(cc)
        container_patch[cavity_r0 + rr, cavity_c0 + cc] = BLACK
        cavity_cells.append((int(cavity_r0 + rr), int(cavity_c0 + cc)))

    piece_patch = np.full(piece_mask.shape, BLACK, dtype=int)
    piece_cells = []
    for rr, cc in np.argwhere(piece_mask):
        rr = int(rr)
        cc = int(cc)
        piece_patch[rr, cc] = color
        piece_cells.append((rr, cc))

    return {
        "color": int(color),
        "required_rotation": int(required_rotation),
        "container_patch": container_patch,
        "piece_patch": piece_patch,
        "cavity_cells": cavity_cells,
        "piece_cells": piece_cells,
    }


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    if top < 1 or left < 1 or top + height >= h or left + width >= w:
        return False
    return not reserved[top - 1 : top + height + 1, left - 1 : left + width + 1].any()


def _mark_reserved(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    reserved[top - 1 : top + height + 1, left - 1 : left + width + 1] = True


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    h, w = patch.shape
    grid_h, grid_w = grid.shape
    if h + 2 > grid_h or w + 2 > grid_w:
        return None

    tries = 120
    for _ in range(tries):
        top = random.randint(1, grid_h - h - 1)
        left = random.randint(1, grid_w - w - 1)
        if not _can_place(reserved, top, left, h, w):
            continue
        grid[top : top + h, left : left + w][patch != BLACK] = patch[patch != BLACK]
        _mark_reserved(reserved, top, left, h, w)
        return int(top), int(left)
    return None


def _gen_attempt(branch_rotation: int) -> np.ndarray | None:
    pair_count = random.choice((1, 2, 2, 3, 3))
    min_side = 12 + 4 * pair_count
    grid_h = random.randint(min_side, 30)
    grid_w = random.randint(min_side, 30)

    colors = random.sample(COLOR_POOL, pair_count)
    specs = []
    for idx, color in enumerate(colors):
        forced = branch_rotation if idx == 0 else None
        specs.append(_build_pair_spec(int(color), forced))

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[tuple[str, int], tuple[int, int]] = {}

    objects = []
    for idx, spec in enumerate(specs):
        objects.append(("container", idx, spec["container_patch"]))
        objects.append(("piece", idx, spec["piece_patch"]))
    objects.sort(key=lambda item: int(item[2].shape[0] * item[2].shape[1]), reverse=True)

    for kind, idx, patch in objects:
        pos = _place_patch(grid, reserved, patch)
        if pos is None:
            return None
        placements[(kind, idx)] = pos

    expected = grid.copy()
    for idx, spec in enumerate(specs):
        piece_top, piece_left = placements[("piece", idx)]
        for rr, cc in spec["piece_cells"]:
            expected[piece_top + int(rr), piece_left + int(cc)] = BLACK

        cont_top, cont_left = placements[("container", idx)]
        for rr, cc in spec["cavity_cells"]:
            expected[cont_top + int(rr), cont_left + int(cc)] = int(spec["color"])

    try:
        containers, pieces = _parse_scene(grid)
    except AssertionError:
        return None
    if len(containers) != pair_count or len(pieces) != pair_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch_rotation = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(branch_rotation)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
