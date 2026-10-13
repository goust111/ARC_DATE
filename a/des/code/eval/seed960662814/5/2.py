from __future__ import annotations

import random

import numpy as np


FILL_COLOR = 9
GEN_BG = 0
GEN_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_PAIR_COUNT_RANGE = (1, 4)
GEN_MAX_SIDE = 24
GEN_MIN_SIDE_BASE = 13
GEN_BORDER_MARGIN = 1
GEN_GAP = 1
GEN_ATTEMPTS = 400

# A 4-cell T tetromino gives a 4-pixel cavity and an 8-pixel minimal shell,
# which keeps both paired objects inside the stated 4..8 pixel size band.
BASE_CAVITY = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
    ],
    dtype=bool,
)


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    raise ValueError(f"unsupported connectivity: {connectivity}")


def _connected_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    neigh = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neigh:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _connected_components(grid != bg, connectivity=8)
    objects: list[dict] = []
    for coords in comps:
        colors = {int(grid[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each object should be monochrome"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": patch == color,
            }
        )
    return objects


def _enclosed_holes(obj: dict, bg: int) -> list[np.ndarray]:
    patch = obj["patch"]
    bg_comps = _connected_components(patch == bg, connectivity=4)
    holes: list[np.ndarray] = []
    h, w = patch.shape
    for comp in bg_comps:
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if not bool(touches_border):
            holes.append(comp)
    return holes


def _match_under_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    if a.sum() != b.sum():
        return False
    for k in range(4):
        if np.array_equal(np.rot90(a, k), b):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, identify the hollow frame and the solid stamp,
    then fill the frame cavity with a fixed maroon color and erase the stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    objects = _extract_objects(g, bg)
    assert objects, "expected at least one foreground object"

    by_color: dict[int, list[dict]] = {}
    for obj in objects:
        by_color.setdefault(int(obj["color"]), []).append(obj)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2, f"color {color} must appear in exactly two objects"
        analyzed = []
        for obj in pair:
            holes = _enclosed_holes(obj, bg)
            analyzed.append((obj, holes))

        frame_items = [(obj, holes) for obj, holes in analyzed if holes]
        solid_items = [(obj, holes) for obj, holes in analyzed if not holes]
        assert len(frame_items) == 1 and len(solid_items) == 1, "each color needs one frame and one solid"

        frame_obj, holes = frame_items[0]
        solid_obj, _ = solid_items[0]
        assert len(holes) == 1, "frame should contain exactly one enclosed cavity"

        cavity = holes[0]
        cavity_mask = _tight_mask(cavity)
        solid_mask = solid_obj["mask"]
        assert _match_under_rotation(solid_mask, cavity_mask), "solid must fit the cavity under rotation"

        for r, c in solid_obj["coords"]:
            out[int(r), int(c)] = bg

        fr0, fc0, _fr1, _fc1 = frame_obj["bbox"]
        for dr, dc in cavity:
            out[int(fr0 + dr), int(fc0 + dc)] = FILL_COLOR

    return out


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask.astype(bool), k)


def _shell_patch(cavity_mask: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    cavity_coords = {(int(r), int(c)) for r, c in np.argwhere(cavity_mask)}
    shell_coords: set[tuple[int, int]] = set()
    for r, c in cavity_coords:
        for dr, dc in _neighbors(4):
            nxt = (r + dr, c + dc)
            if nxt not in cavity_coords:
                shell_coords.add(nxt)

    all_coords = shell_coords | cavity_coords
    r0 = min(r for r, _ in all_coords)
    c0 = min(c for _, c in all_coords)
    r1 = max(r for r, _ in all_coords)
    c1 = max(c for _, c in all_coords)

    frame = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    cavity_in_frame = np.zeros_like(frame, dtype=bool)
    for r, c in shell_coords:
        frame[r - r0, c - c0] = color
    for r, c in cavity_coords:
        cavity_in_frame[r - r0, c - c0] = True
    return frame, cavity_in_frame


def _piece_patch(cavity_mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(cavity_mask.shape, bg, dtype=int)
    patch[cavity_mask] = color
    return patch


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int, gap: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < GEN_BORDER_MARGIN or left < GEN_BORDER_MARGIN:
        return False
    if top + h > gh - GEN_BORDER_MARGIN or left + w > gw - GEN_BORDER_MARGIN:
        return False
    fg = np.argwhere(patch != bg)
    for dr, dc in fg:
        r = int(top + dr)
        c = int(left + dc)
        r0 = max(0, r - gap)
        c0 = max(0, c - gap)
        r1 = min(gh, r + gap + 1)
        c1 = min(gw, c + gap + 1)
        if np.any(grid[r0:r1, c0:c1] != bg):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> np.ndarray:
    out = grid.copy()
    h, w = patch.shape
    target = out[top : top + h, left : left + w]
    mask = patch != bg
    target[mask] = patch[mask]
    return out


def _all_positions(grid: np.ndarray, patch: np.ndarray, bg: int, gap: int) -> list[tuple[int, int]]:
    gh, gw = grid.shape
    h, w = patch.shape
    positions = []
    for top in range(0, gh - h + 1):
        for left in range(0, gw - w + 1):
            if _can_place(grid, patch, top, left, bg, gap):
                positions.append((top, left))
    return positions


def _build_pair(color: int) -> dict:
    frame_rot = random.randrange(4)
    piece_rot = random.randrange(4)
    cavity_for_frame = _rotate_mask(BASE_CAVITY, frame_rot)
    cavity_for_piece = _rotate_mask(BASE_CAVITY, piece_rot)
    frame_patch, cavity_in_frame = _shell_patch(cavity_for_frame, color, GEN_BG)
    piece_patch = _piece_patch(cavity_for_piece, color, GEN_BG)
    return {
        "color": int(color),
        "frame_patch": frame_patch,
        "piece_patch": piece_patch,
        "cavity_in_frame": cavity_in_frame,
    }


def _attempt_example() -> tuple[np.ndarray, np.ndarray] | None:
    n_pairs = random.randint(*GEN_PAIR_COUNT_RANGE)
    min_side = GEN_MIN_SIDE_BASE + 2 * n_pairs
    height = random.randint(min_side, GEN_MAX_SIDE)
    width = random.randint(min_side, GEN_MAX_SIDE)

    colors = random.sample(GEN_COLOR_POOL, n_pairs)
    pairs = [_build_pair(color) for color in colors]

    grid = np.full((height, width), GEN_BG, dtype=int)
    placements: list[dict] = []

    for pair in pairs:
        frame_positions = _all_positions(grid, pair["frame_patch"], GEN_BG, GEN_GAP)
        if not frame_positions:
            return None
        frame_top, frame_left = random.choice(frame_positions)
        grid = _place_patch(grid, pair["frame_patch"], frame_top, frame_left, GEN_BG)

        piece_positions = _all_positions(grid, pair["piece_patch"], GEN_BG, GEN_GAP)
        if not piece_positions:
            return None
        piece_top, piece_left = random.choice(piece_positions)
        grid = _place_patch(grid, pair["piece_patch"], piece_top, piece_left, GEN_BG)

        cavity_abs = np.argwhere(pair["cavity_in_frame"])
        cavity_abs[:, 0] += frame_top
        cavity_abs[:, 1] += frame_left
        piece_abs = np.argwhere(pair["piece_patch"] != GEN_BG)
        piece_abs[:, 0] += piece_top
        piece_abs[:, 1] += piece_left

        placements.append(
            {
                "cavity_abs": cavity_abs,
                "piece_abs": piece_abs,
            }
        )

    expected = grid.copy()
    for placement in placements:
        for r, c in placement["piece_abs"]:
            expected[int(r), int(c)] = GEN_BG
        for r, c in placement["cavity_abs"]:
            expected[int(r), int(c)] = FILL_COLOR

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        example = _attempt_example()
        if example is not None:
            return example[0]
    raise RuntimeError("failed to generate a valid input")
