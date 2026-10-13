from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
HUB_SIZE = 5
CONTACT_LEN = HUB_SIZE - 2
HUB_COLORS = {"top": 1, "bottom": 2, "left": 3, "right": 4}
SPRITE_COLORS = (5, 6, 7)
SIDES = ("top", "bottom", "left", "right")
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

CANONICAL_TOP_TEMPLATES = [
    ["..#..", ".###.", "##.##", ".###."],
    [".#.#.", "#####", ".###.", ".###."],
    ["..#..", "#####", ".###.", ".###."],
    [".#.", "###", ".#.", "###", "###"],
    [".#.#.", "#####", ".###.", ".#.#.", ".###."],
]


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    rr = slice(int(r0), int(r0 + h))
    cc = slice(int(c0), int(c0 + w))
    view = dst[rr, cc]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components(mask: np.ndarray, *, connectivity: int) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    neigh = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in neigh:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _paint_hub(canvas: np.ndarray, hub_r: int, hub_c: int) -> None:
    canvas[hub_r, hub_c + 1 : hub_c + 4] = HUB_COLORS["top"]
    canvas[hub_r + 4, hub_c + 1 : hub_c + 4] = HUB_COLORS["bottom"]
    canvas[hub_r + 1 : hub_r + 4, hub_c] = HUB_COLORS["left"]
    canvas[hub_r + 1 : hub_r + 4, hub_c + 4] = HUB_COLORS["right"]


def _find_hub(grid: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = grid.shape
    center = ((H - 1) / 2.0, (W - 1) / 2.0)
    best: tuple[float, tuple[int, int]] | None = None
    for r in range(H - HUB_SIZE + 1):
        for c in range(W - HUB_SIZE + 1):
            patch = grid[r : r + HUB_SIZE, c : c + HUB_SIZE]
            if int(patch[0, 0]) != bg or int(patch[0, 4]) != bg or int(patch[4, 0]) != bg or int(patch[4, 4]) != bg:
                continue
            if not np.all(patch[1:4, 1:4] == bg):
                continue
            if not np.all(patch[0, 1:4] == HUB_COLORS["top"]):
                continue
            if not np.all(patch[4, 1:4] == HUB_COLORS["bottom"]):
                continue
            if not np.all(patch[1:4, 0] == HUB_COLORS["left"]):
                continue
            if not np.all(patch[1:4, 4] == HUB_COLORS["right"]):
                continue
            score = abs((r + 2) - center[0]) + abs((c + 2) - center[1])
            if best is None or score < best[0]:
                best = (float(score), (int(r), int(c)))
    assert best is not None, "expected one central 5x5 hollow hub"
    return best[1]


def _boundary_marker_pairs(grid: np.ndarray, bg: int, hub_r: int, hub_c: int) -> dict[int, dict[str, int | str]]:
    H, W = grid.shape
    center_r = hub_r + 2
    center_c = hub_c + 2
    pairs: dict[int, dict[str, int | str]] = {}
    for color in SPRITE_COLORS:
        coords = []
        for c in range(W):
            if int(grid[0, c]) == color:
                coords.append((0, c))
            if int(grid[H - 1, c]) == color:
                coords.append((H - 1, c))
        for r in range(1, H - 1):
            if int(grid[r, 0]) == color:
                coords.append((r, 0))
            if int(grid[r, W - 1]) == color:
                coords.append((r, W - 1))
        if not coords:
            continue
        assert len(coords) == 2, "each used sprite color must have exactly two boundary markers"
        (r1, c1), (r2, c2) = coords
        if r1 == 0 and r2 == H - 1 and c1 == c2:
            col = int(c1)
            assert np.all(grid[1 : H - 1, col] == bg), "vertical corridor must be unobstructed"
            side = "left" if col < center_c else "right"
            pairs[color] = {"side": side, "axis": "vertical", "index": col}
        elif c1 == 0 and c2 == W - 1 and r1 == r2:
            row = int(r1)
            assert np.all(grid[row, 1 : W - 1] == bg), "horizontal corridor must be unobstructed"
            side = "top" if row < center_r else "bottom"
            pairs[color] = {"side": side, "axis": "horizontal", "index": row}
        else:
            raise AssertionError("markers must lie on opposite edges of the same row or column")
    assert len(pairs) >= 2, "expected 2-3 corridor colors"
    return pairs


def _extract_sprite_mask(grid: np.ndarray, color: int) -> np.ndarray:
    H, W = grid.shape
    mask = grid == int(color)
    mask[0, :] = False
    mask[H - 1, :] = False
    mask[:, 0] = False
    mask[:, W - 1] = False
    comps = _components(mask, connectivity=8)
    assert len(comps) == 1, "each sprite color should form exactly one interior sprite"
    local = np.zeros_like(mask, dtype=bool)
    local[comps[0][:, 0], comps[0][:, 1]] = True
    return _crop_mask(local)


def _edge_indices(mask: np.ndarray, side: str) -> list[int]:
    if side == "top":
        return list(np.flatnonzero(mask[-1]))
    if side == "bottom":
        return list(np.flatnonzero(mask[0]))
    if side == "left":
        return list(np.flatnonzero(mask[:, -1]))
    assert side == "right"
    return list(np.flatnonzero(mask[:, 0]))


def _indices_are_contiguous(values: list[int]) -> bool:
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def _mirror_matches_side(mask: np.ndarray, side: str) -> bool:
    # Use the same disambiguation style as the docking reference:
    # top/bottom attachments need a vertical mirror axis; left/right need a horizontal one.
    if side in ("top", "bottom"):
        return bool(np.array_equal(mask, np.fliplr(mask)))
    return bool(np.array_equal(mask, np.flipud(mask)))


def _valid_rotations(mask: np.ndarray, side: str) -> list[tuple[int, np.ndarray]]:
    valid: list[tuple[int, np.ndarray]] = []
    for k in range(4):
        rotated = _crop_mask(np.rot90(mask, k))
        edge = _edge_indices(rotated, side)
        if len(edge) != CONTACT_LEN or not _indices_are_contiguous(edge):
            continue
        if not _mirror_matches_side(rotated, side):
            continue
        valid.append((k, rotated))
    return valid


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _attached_top_left(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    edge = _edge_indices(mask, side)
    if side == "top":
        return int(hub_r - mask.shape[0]), int(hub_c + 1 - edge[0])
    if side == "bottom":
        return int(hub_r + HUB_SIZE), int(hub_c + 1 - edge[0])
    if side == "left":
        return int(hub_r + 1 - edge[0]), int(hub_c - mask.shape[1])
    return int(hub_r + 1 - edge[0]), int(hub_c + HUB_SIZE)


def _attach_sprite(out: np.ndarray, mask: np.ndarray, color: int, side: str, hub_r: int, hub_c: int) -> None:
    r0, c0 = _attached_top_left(mask, side, hub_r, hub_c)
    _paste(out, _color_patch(mask, color), r0, c0, transparent=BG)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach each monochrome sprite to the hub side indicated by its border corridor.

    Input structure:
    - The background is black. Near the center sits a 5x5 hollow hub whose four sides have fixed colors and whose corners/interior stay black.
    - There are 2-3 scattered monochrome sprites in colors {5,6,7}.
    - For every sprite color, the outer border contains exactly two same-colored marker pixels on opposite edges, and the row or column between them is an unobstructed black corridor.

    Rule:
    - Detect the hub, then pair same-colored border markers to recover each corridor.
    - A vertical corridor (top-to-bottom markers) selects the hub's left or right side according to whether its column lies left or right of the hub center; a horizontal corridor (left-to-right markers) similarly selects the top or bottom side from its row.
    - Find the same-colored sprite, rotate it to the unique orientation whose attachment edge has exactly three contiguous contact pixels and whose mirror symmetry matches that side, then translate it flush against the chosen hub side.
    - Output only the hub plus the attached sprites; clear the original scattered sprites and the border markers.

    Key insight:
    - Color no longer matches sprite-to-hub sides; the border corridor geometry determines the target side, and that side then parameterizes the rotation-and-docking step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _mode_color(g)
    hub_r, hub_c = _find_hub(g, bg)
    side_info = _boundary_marker_pairs(g, bg, hub_r, hub_c)

    out = np.full_like(g, bg)
    _paint_hub(out, hub_r, hub_c)

    for color, info in side_info.items():
        side = str(info["side"])
        sprite = _extract_sprite_mask(g, color)
        candidates = _valid_rotations(sprite, side)
        assert len(candidates) == 1, "sprite rotation must be uniquely determined"
        _k, rotated = candidates[0]
        _attach_sprite(out, rotated, color, side, hub_r, hub_c)

    return out


def _axes(mask: np.ndarray) -> set[str]:
    axes = set()
    if np.array_equal(mask, np.flipud(mask)):
        axes.add("h")
    if np.array_equal(mask, np.fliplr(mask)):
        axes.add("v")
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            axes.add("d")
        if np.array_equal(mask, np.flipud(np.fliplr(mask.T))):
            axes.add("a")
    return axes


def _template_bank() -> list[np.ndarray]:
    bank = [_mask_from_rows(rows) for rows in CANONICAL_TOP_TEMPLATES]
    for mask in bank:
        assert _axes(mask) == {"v"}, "templates must expose exactly one reflection axis"
        assert len(_valid_rotations(mask, "top")) == 1
    return bank


TEMPLATE_BANK = _template_bank()


def _build_output(H: int, W: int, hub_r: int, hub_c: int, side_to_sprite: dict[str, tuple[int, np.ndarray]]) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    _paint_hub(out, hub_r, hub_c)
    for side, (color, mask) in side_to_sprite.items():
        _attach_sprite(out, mask, color, side, hub_r, hub_c)
    return out


def _sample_hub_origin(length: int) -> int:
    center = length // 2 - 2
    lo = max(5, center - 1)
    hi = min(length - 10, center + 1)
    return random.randint(lo, hi)


def _choose_sides() -> list[str]:
    count = random.choice([2, 3])
    sides = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    if count == 3:
        leftovers = [side for side in SIDES if side not in sides]
        sides.append(random.choice(leftovers))
    random.shuffle(sides)
    return sides


def _choose_corridors(H: int, W: int, hub_r: int, hub_c: int, sides: list[str]) -> dict[str, int]:
    chosen: dict[str, int] = {}
    if "top" in sides:
        chosen["top"] = random.randint(1, hub_r - 2)
    if "bottom" in sides:
        chosen["bottom"] = random.randint(hub_r + HUB_SIZE, H - 2)
    if "left" in sides:
        chosen["left"] = random.randint(1, hub_c - 2)
    if "right" in sides:
        chosen["right"] = random.randint(hub_c + HUB_SIZE, W - 2)
    return chosen


def _fits(mask: np.ndarray, top: int, left: int, H: int, W: int) -> bool:
    return 0 <= top and 0 <= left and top + mask.shape[0] <= H and left + mask.shape[1] <= W


def _patch_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in np.argwhere(mask)]


def _far_enough(coords: list[tuple[int, int]], occupied: set[tuple[int, int]], *, gap: int) -> bool:
    for r, c in coords:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_scattered_sprite(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: int,
    forbidden_rows: set[int],
    forbidden_cols: set[int],
    occupied: set[tuple[int, int]],
) -> None:
    H, W = canvas.shape
    for _ in range(400):
        top = random.randint(1, H - mask.shape[0] - 1)
        left = random.randint(1, W - mask.shape[1] - 1)
        if not _fits(mask, top, left, H, W):
            continue
        coords = _patch_coords(mask, top, left)
        if any(r in forbidden_rows or c in forbidden_cols for r, c in coords):
            continue
        if not _far_enough(coords, occupied, gap=2):
            continue
        canvas[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = int(color)
        occupied.update(coords)
        return
    raise AssertionError("failed to place scattered sprite")


def _marker_cells(side: str, index: int, H: int, W: int) -> list[tuple[int, int]]:
    if side in ("left", "right"):
        return [(0, index), (H - 1, index)]
    return [(index, 0), (index, W - 1)]


def _analyze_output_sprites(out: np.ndarray, hub_r: int, hub_c: int) -> dict[int, dict[str, object]]:
    mask = out != BG
    hub_mask = np.zeros_like(mask, dtype=bool)
    hub_mask[hub_r : hub_r + HUB_SIZE, hub_c : hub_c + HUB_SIZE] = True
    sprite_mask = mask & (~hub_mask)
    comps = _components(sprite_mask, connectivity=8)
    info: dict[int, dict[str, object]] = {}
    for comp in comps:
        colors = {int(out[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        local = np.zeros_like(sprite_mask, dtype=bool)
        local[comp[:, 0], comp[:, 1]] = True
        box = _bbox(local)
        assert box is not None
        r0, c0, r1, c1 = box
        cropped = local[r0 : r1 + 1, c0 : c1 + 1]
        if r1 == hub_r - 1:
            side = "top"
        elif r0 == hub_r + HUB_SIZE:
            side = "bottom"
        elif c1 == hub_c - 1:
            side = "left"
        elif c0 == hub_c + HUB_SIZE:
            side = "right"
        else:
            raise AssertionError("attached sprite must touch exactly one hub side")
        info[color] = {"side": side, "mask": cropped}
    return info


def _validate_instance(inp: np.ndarray, expected_out: np.ndarray, color_to_side: dict[int, str]) -> None:
    produced = main(inp)
    assert np.array_equal(produced, expected_out), "solver output must match constructed target"

    hub_r, hub_c = _find_hub(inp, BG)
    marker_info = _boundary_marker_pairs(inp, BG, hub_r, hub_c)
    assert {int(color): str(info["side"]) for color, info in marker_info.items()} == color_to_side

    attached = _analyze_output_sprites(expected_out, hub_r, hub_c)
    assert set(attached) == set(color_to_side)
    for color, side in color_to_side.items():
        sprite = _extract_sprite_mask(inp, color)
        assert len(_valid_rotations(sprite, side)) == 1
        assert str(attached[color]["side"]) == side
        assert np.array_equal(_valid_rotations(sprite, side)[0][1], attached[color]["mask"])

    H, W = inp.shape
    for color, side in color_to_side.items():
        markers = _marker_cells(side, int(marker_info[color]["index"]), H, W)
        for r, c in markers:
            assert int(inp[r, c]) == color


def generate_input() -> np.ndarray:
    for _ in range(300):
        H = random.randint(17, 23)
        W = random.randint(17, 23)
        hub_r = _sample_hub_origin(H)
        hub_c = _sample_hub_origin(W)
        sides = _choose_sides()
        colors = list(random.sample(SPRITE_COLORS, k=len(sides)))
        random.shuffle(colors)
        color_to_side = dict(zip(colors, sides))
        corridors = _choose_corridors(H, W, hub_r, hub_c, sides)

        side_to_sprite: dict[str, tuple[int, np.ndarray]] = {}
        for color, side in color_to_side.items():
            base = random.choice(TEMPLATE_BANK)
            candidates = _valid_rotations(base, side)
            assert len(candidates) == 1
            side_to_sprite[side] = (color, candidates[0][1])

        expected_out = _build_output(H, W, hub_r, hub_c, side_to_sprite)

        inp = np.zeros((H, W), dtype=int)
        _paint_hub(inp, hub_r, hub_c)

        occupied = set((int(r), int(c)) for r in range(hub_r, hub_r + HUB_SIZE) for c in range(hub_c, hub_c + HUB_SIZE))
        forbidden_rows = {corridors[side] for side in ("top", "bottom") if side in corridors}
        forbidden_cols = {corridors[side] for side in ("left", "right") if side in corridors}

        try:
            for color, side in color_to_side.items():
                final_mask = side_to_sprite[side][1]
                k = random.choice([1, 2, 3])
                scattered = _crop_mask(np.rot90(final_mask, k))
                _place_scattered_sprite(inp, scattered, color, forbidden_rows, forbidden_cols, occupied)
        except AssertionError:
            continue

        for color, side in color_to_side.items():
            for r, c in _marker_cells(side, corridors[side], H, W):
                inp[r, c] = int(color)

        try:
            _validate_instance(inp, expected_out, color_to_side)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid instance")
