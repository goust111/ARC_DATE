from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox, crop_bbox, paste
except Exception:
    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        h, w = patch.shape
        if not clip:
            if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
                raise ValueError("paste out of bounds")
            rr0, cc0, rr1, cc1 = r0, c0, r0 + h, c0 + w
            pr0, pc0 = 0, 0
        else:
            rr0 = max(0, int(r0))
            cc0 = max(0, int(c0))
            rr1 = min(dst.shape[0], int(r0) + h)
            cc1 = min(dst.shape[1], int(c0) + w)
            if rr0 >= rr1 or cc0 >= cc1:
                return dst
            pr0, pc0 = rr0 - int(r0), cc0 - int(c0)
        pr1 = pr0 + (rr1 - rr0)
        pc1 = pc0 + (cc1 - cc0)
        view = patch[pr0:pr1, pc0:pc1]
        if transparent is None:
            dst[rr0:rr1, cc0:cc1] = view
        else:
            m = view != int(transparent)
            dst_view = dst[rr0:rr1, cc0:cc1]
            dst_view[m] = view[m]
        return dst


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5

CELL = 5
CELL_GAP = 1
REGION = CELL * 2 + CELL_GAP
LEFT_TOP = 1
LEFT_LEFT = 1
GAP_TO_DIVIDER = 1
GAP_FROM_DIVIDER = 1
SOURCE_DELTA = {"ibeam7": 1, "plus5": 2, "rect6": 3}
LEFT_CELL_BY_SHAPE = {"ibeam7": (0, 0), "plus5": (0, 1), "rect6": (1, 0)}
BOARD_CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]
MAX_ATTEMPTS = 400


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_signature(mask: np.ndarray) -> bytes:
    tight = crop_bbox(mask.astype(int), bbox(mask))
    return tight.astype(np.uint8).tobytes() + str(tight.shape).encode()


def _marker_side(sprite: np.ndarray) -> str:
    locs = np.argwhere(sprite == BLUE)
    assert len(locs) == 1
    r, c = map(int, locs[0])
    h, w = sprite.shape
    sides = []
    if r == 0:
        sides.append("N")
    if r == h - 1:
        sides.append("S")
    if c == 0:
        sides.append("W")
    if c == w - 1:
        sides.append("E")
    assert len(sides) == 1
    return sides[0]


def _indicator_side(slot_coords: np.ndarray, indicator: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = _bbox_from_coords(slot_coords)
    rr, cc = map(int, indicator)
    slot_set = {tuple(map(int, xy)) for xy in slot_coords}
    if rr == r0 - 1 and (rr + 1, cc) in slot_set:
        return "N"
    if rr == r1 + 1 and (rr - 1, cc) in slot_set:
        return "S"
    if cc == c0 - 1 and (rr, cc + 1) in slot_set:
        return "W"
    if cc == c1 + 1 and (rr, cc - 1) in slot_set:
        return "E"
    return None


def _board_cell_origin(board_top: int, board_left: int, cell: tuple[int, int]) -> tuple[int, int]:
    cr, cc = cell
    return (
        int(board_top + cr * (CELL + CELL_GAP)),
        int(board_left + cc * (CELL + CELL_GAP)),
    )


def _slot_offsets(sprite: np.ndarray) -> tuple[int, int]:
    h, w = sprite.shape
    return (CELL - h) // 2, (CELL - w) // 2


def _shape_candidates(sprite: np.ndarray, slot_mask: np.ndarray) -> list[int]:
    sprite_mask = sprite != BLACK
    ks = []
    for k in range(4):
        turned = np.rot90(sprite_mask, k)
        if turned.shape == slot_mask.shape and np.array_equal(turned, slot_mask):
            ks.append(k)
    return ks


def _unique_matchings(edges: list[list[int]], n_slots: int) -> list[list[int]]:
    order = sorted(range(len(edges)), key=lambda idx: len(edges[idx]))
    used = [False] * n_slots
    partial = [-1] * len(edges)
    out: list[list[int]] = []

    def dfs(pos: int) -> None:
        if len(out) > 1:
            return
        if pos == len(order):
            out.append(partial.copy())
            return
        s_idx = order[pos]
        for t_idx in edges[s_idx]:
            if used[t_idx]:
                continue
            used[t_idx] = True
            partial[s_idx] = t_idx
            dfs(pos + 1)
            partial[s_idx] = -1
            used[t_idx] = False

    dfs(0)
    return out


def _extract_sprites(grid: np.ndarray, divider_c: int) -> list[dict]:
    left = grid[:, :divider_c]
    comps = _component_coords(np.isin(left, [BLUE, GREEN]))
    sprites = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        crop = crop_bbox(left, (r0, c0, r1, c1))
        assert 4 <= int(np.count_nonzero(crop)) <= 7
        assert int(np.sum(crop == BLUE)) == 1
        assert np.all(np.isin(crop[crop != BLACK], [BLUE, GREEN]))
        assert _marker_side(crop) in {"N", "E", "S", "W"}
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "crop": crop,
                "coords_abs": [(int(r), int(c)) for r, c in coords],
            }
        )
    sprites.sort(key=lambda item: item["bbox"])
    assert 2 <= len(sprites) <= 3
    return sprites


def _extract_slots(grid: np.ndarray, divider_c: int) -> tuple[tuple[int, int, int, int], list[dict]]:
    right = grid[:, divider_c + 1 :]
    board_box = bbox(right == YELLOW)
    assert board_box is not None
    br0, bc0, br1, bc1 = board_box
    board = crop_bbox(right, board_box)
    slot_comps = _component_coords(board == BLACK)
    blue_cells = [tuple(map(int, xy)) for xy in np.argwhere(board == BLUE)]
    slots = []
    used_blues: set[tuple[int, int]] = set()
    for coords in slot_comps:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for rr, cc in coords:
            mask[int(rr) - r0, int(cc) - c0] = True
        matches = []
        for blue in blue_cells:
            side = _indicator_side(coords, blue)
            if side is not None:
                matches.append((blue, side))
        assert len(matches) == 1
        used_blues.add(matches[0][0])
        abs_r0 = int(br0 + r0)
        abs_c0 = int(divider_c + 1 + bc0 + c0)
        slots.append(
            {
                "bbox_abs": (abs_r0, abs_c0, abs_r0 + mask.shape[0] - 1, abs_c0 + mask.shape[1] - 1),
                "mask": mask,
                "indicator_side": matches[0][1],
            }
        )
    assert used_blues == set(blue_cells)
    slots.sort(key=lambda item: item["bbox_abs"])
    assert 2 <= len(slots) <= 3
    return (
        int(br0),
        int(divider_c + 1 + bc0),
        int(br1),
        int(divider_c + 1 + bc1),
    ), slots


def _solve(grid: np.ndarray) -> dict:
    divider_cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(divider_cols) == 1
    divider_c = divider_cols[0]
    sprites = _extract_sprites(grid, divider_c)
    board_box, slots = _extract_slots(grid, divider_c)
    assert len(sprites) == len(slots)

    candidate_ks: list[list[list[int]]] = []
    edges: list[list[int]] = []
    for sprite in sprites:
        row = []
        edge_row = []
        for idx, slot in enumerate(slots):
            ks = _shape_candidates(sprite["crop"], slot["mask"])
            row.append(ks)
            if ks:
                edge_row.append(idx)
        candidate_ks.append(row)
        edges.append(edge_row)

    matchings = _unique_matchings(edges, len(slots))
    assert len(matchings) == 1
    pairing = matchings[0]

    placements = []
    for s_idx, t_idx in enumerate(pairing):
        sprite = sprites[s_idx]
        slot = slots[t_idx]
        shape_ks = candidate_ks[s_idx][t_idx]
        assert len(shape_ks) >= 2
        orient_ks = [k for k in shape_ks if _marker_side(np.rot90(sprite["crop"], k)) == slot["indicator_side"]]
        assert len(orient_ks) == 1
        placements.append(
            {
                "sprite_index": s_idx,
                "slot_index": t_idx,
                "rotation": orient_ks[0],
                "shape_candidates": shape_ks,
            }
        )

    return {
        "divider_c": divider_c,
        "board_box": board_box,
        "sprites": sprites,
        "slots": slots,
        "placements": placements,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate left-side marked sprites into their matching right-side slots, using the blue marker to resolve rotation.

    Input structure:
    - A black background is split by one full-height grey vertical divider.
    - Left of the divider are 2-3 disconnected green sprites, each containing exactly one blue pixel.
    - Right of the divider is one yellow rectangular board containing the same number of black slot holes; each slot has one adjacent blue indicator pixel inside the board.

    Rule:
    - For each left sprite, compare the union of its green and blue cells against every slot under the four rotations to find the unique slot pairing by shape compatibility.
    - Because the sprites are rotation-symmetric, shape matching leaves multiple candidate rotations; keep only the rotation whose blue sprite pixel lies on the same bbox side (north/east/south/west) as the slot's blue indicator.
    - Erase the original left sprites and paste the rotated sprite into the matched slot, while keeping the grey divider, yellow board, and blue indicator pixels unchanged.

    Key insight:
    - Shape compatibility decides which sprite belongs to which slot, but only the blue marker-versus-indicator side correspondence removes the remaining rotational ambiguity.
    """
    g = np.asarray(grid, dtype=int)
    info = _solve(g)
    out = g.copy()

    for sprite in info["sprites"]:
        for r, c in sprite["coords_abs"]:
            out[int(r), int(c)] = BLACK

    for placement in info["placements"]:
        sprite = info["sprites"][placement["sprite_index"]]
        slot = info["slots"][placement["slot_index"]]
        turned = np.rot90(sprite["crop"], placement["rotation"])
        r0, c0, _r1, _c1 = slot["bbox_abs"]
        paste(out, turned, r0, c0, transparent=BLACK)

    return out


def _base_patches() -> dict[str, np.ndarray]:
    plus = np.array(
        [
            [0, 1, 0],
            [3, 3, 3],
            [0, 3, 0],
        ],
        dtype=int,
    )
    rect = np.array(
        [
            [3, 1, 3],
            [3, 3, 3],
        ],
        dtype=int,
    )
    ibeam = np.array(
        [
            [3, 1, 3],
            [0, 3, 0],
            [3, 3, 3],
        ],
        dtype=int,
    )
    return {"plus5": plus, "rect6": rect, "ibeam7": ibeam}


BASE_PATCHES = _base_patches()


def _left_origin_for_shape(shape_key: str) -> tuple[int, int]:
    cell = LEFT_CELL_BY_SHAPE[shape_key]
    cell_r = LEFT_TOP + cell[0] * (CELL + CELL_GAP)
    cell_c = LEFT_LEFT + cell[1] * (CELL + CELL_GAP)
    sprite = BASE_PATCHES[shape_key]
    off_r, off_c = _slot_offsets(sprite)
    return cell_r + off_r, cell_c + off_c


def _make_target_patch(shape_key: str) -> np.ndarray:
    return np.rot90(BASE_PATCHES[shape_key], random.randrange(4))


def _paint_slot(board: np.ndarray, cell: tuple[int, int], target_patch: np.ndarray) -> None:
    cell_r, cell_c = _board_cell_origin(0, 0, cell)
    off_r, off_c = _slot_offsets(target_patch)
    slot_r = cell_r + off_r
    slot_c = cell_c + off_c
    mask = target_patch != BLACK
    board[slot_r : slot_r + target_patch.shape[0], slot_c : slot_c + target_patch.shape[1]][mask] = BLACK

    marker_r, marker_c = map(int, np.argwhere(target_patch == BLUE)[0])
    side = _marker_side(target_patch)
    if side == "N":
        board[slot_r - 1, slot_c + marker_c] = BLUE
    elif side == "S":
        board[slot_r + target_patch.shape[0], slot_c + marker_c] = BLUE
    elif side == "W":
        board[slot_r + marker_r, slot_c - 1] = BLUE
    else:
        board[slot_r + marker_r, slot_c + target_patch.shape[1]] = BLUE


def _build_instance() -> np.ndarray:
    count = random.choice([2, 3])
    shape_keys = sorted(random.sample(sorted(BASE_PATCHES), count))
    board_cells = random.sample(BOARD_CELLS, count)
    random.shuffle(board_cells)

    board = np.full((REGION, REGION), YELLOW, dtype=int)
    target_patches: dict[str, np.ndarray] = {}
    for shape_key, cell in zip(shape_keys, board_cells):
        target_patch = _make_target_patch(shape_key)
        target_patches[shape_key] = target_patch
        _paint_slot(board, cell, target_patch)

    board_top = random.randint(1, 4)
    divider_c = LEFT_LEFT + REGION + GAP_TO_DIVIDER
    board_left = divider_c + 1 + GAP_FROM_DIVIDER
    right_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    h = max(LEFT_TOP + REGION, board_top + REGION) + bottom_margin
    w = board_left + REGION + right_margin

    grid = np.zeros((h, w), dtype=int)
    grid[:, divider_c] = GREY
    paste(grid, board, board_top, board_left)

    for shape_key in shape_keys:
        target_patch = target_patches[shape_key]
        source_patch = np.rot90(target_patch, SOURCE_DELTA[shape_key])
        r0, c0 = _left_origin_for_shape(shape_key)
        paste(grid, source_patch, r0, c0, transparent=BLACK)

    return grid


def _instance_semantics_ok(grid: np.ndarray) -> bool:
    try:
        info = _solve(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if not (grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30):
        return False
    if len(info["sprites"]) != len(info["slots"]):
        return False
    if len({_shape_signature(slot["mask"]) for slot in info["slots"]}) != len(info["slots"]):
        return False
    if any(len(item["shape_candidates"]) < 2 for item in info["placements"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_instance()
        if _instance_semantics_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
