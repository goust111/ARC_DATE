import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg


_BG = 0
_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_CORNERS = ("tl", "tr", "br", "bl")
_GEN_RETRIES = 400
_GRID_SIZE_RANGE = (13, 18)
_COLOR_COUNT_RANGE = (3, 4)
_COMPONENTS_PER_DISTRACTOR_COLOR = (1, 3)
_VALID_K_CHOICES = (2, 3)


def _corner_l_mask(k: int, corner: str) -> np.ndarray:
    mask = np.zeros((int(k), int(k)), dtype=bool)
    if corner == "tl":
        mask[0, :] = True
        mask[:, 0] = True
    elif corner == "tr":
        mask[0, :] = True
        mask[:, -1] = True
    elif corner == "br":
        mask[-1, :] = True
        mask[:, -1] = True
    elif corner == "bl":
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(f"unknown corner {corner}")
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(mask.shape), tuple(int(x) for x in mask.ravel())


def _component_crop_mask(component: dict) -> np.ndarray:
    coords = component["coords"]
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _detect_corner_l(mask: np.ndarray) -> tuple[int, str]:
    h, w = map(int, mask.shape)
    assert h == w and h >= 2, "selected fragment must crop to a square corner-L"
    for corner in _CORNERS:
        if np.array_equal(mask, _corner_l_mask(h, corner)):
            return h, corner
    raise AssertionError("selected fragment must be a corner-L")


def _rotate_to_tl(mask: np.ndarray, corner: str) -> np.ndarray:
    target = _corner_l_mask(mask.shape[0], "tl")
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        if np.array_equal(rotated, target):
            return rotated
    raise AssertionError(f"cannot normalize corner-L from {corner} to tl")


def _complete_fourfold(mask: np.ndarray) -> np.ndarray:
    k, corner = _detect_corner_l(mask)
    tl = _rotate_to_tl(mask, corner)
    side = 2 * k - 1
    out = np.zeros((side, side), dtype=bool)
    out[:k, :k] |= tl
    out[:k, side - k :] |= np.rot90(tl, 3)
    out[side - k :, side - k :] |= np.rot90(tl, 2)
    out[side - k :, :k] |= np.rot90(tl, 1)
    return out


def _center_mask(shape: tuple[int, int], mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    h, w = map(int, shape)
    side = int(mask.shape[0])
    assert mask.shape[0] == mask.shape[1]
    assert side <= h and side <= w
    out = np.full((h, w), int(bg), dtype=int)
    r0 = (h - side) // 2
    c0 = (w - side) // 2
    out[r0 : r0 + side, c0 : c0 + side][mask] = int(color)
    return out


def _select_repeated_color(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    components = find_components(
        grid,
        bg=int(bg),
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected foreground fragments"

    by_color: dict[int, list[np.ndarray]] = {}
    for component in components:
        color = int(component["color"])
        by_color.setdefault(color, []).append(_component_crop_mask(component))

    candidates: list[tuple[int, np.ndarray]] = []
    for color, masks in by_color.items():
        if len(masks) != 2:
            continue
        keys = [_mask_key(mask) for mask in masks]
        if keys[0] == keys[1]:
            candidates.append((int(color), masks[0]))

    assert len(candidates) == 1, "expected exactly one repeated color fragment"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the only color whose corner-L fragment appears twice, then complete it to a centered fourfold-symmetric motif.

    Input structure:
    - The grid has a black background with 3-4 foreground colors scattered as separate 4-connected fragments.
    - Each fragment is a small monochromatic corner-L crop; colors may have 1-3 components.
    - Exactly one color has exactly two components whose tight cropped patterns are identical.

    Rule:
    - Split the non-background pixels into monochromatic 4-connected components and crop each component to its tight bounding box.
    - Find the unique color whose two cropped component masks are exactly the same, and ignore every other color.
    - Treat that repeated crop as one corner of a larger motif, place rotated copies in all four corners of the minimal odd square it implies, and center the completed motif on a same-sized black output canvas.

    Key insight:
    The repeated pattern only identifies which fragment color survives; the final output is obtained by geometric completion of that selected corner-L, not by returning the fragment itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    color, fragment = _select_repeated_color(g, bg)
    completed = _complete_fourfold(fragment)
    return _center_mask(g.shape, completed, color, bg)


def _catalog() -> list[tuple[int, str, np.ndarray]]:
    items = []
    for k in _VALID_K_CHOICES:
        for corner in _CORNERS:
            items.append((k, corner, _corner_l_mask(k, corner)))
    return items


def _place_patch(
    grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, color: int
) -> bool:
    h, w = map(int, patch.shape)
    gh, gw = map(int, grid.shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            if blocked[r0 : r0 + h, c0 : c0 + w].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w][patch] = int(color)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(gh, r0 + h + 1)
    cc1 = min(gw, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True
    return True


def _build_expected(shape: tuple[int, int], color: int, fragment: np.ndarray) -> np.ndarray:
    completed = _complete_fourfold(fragment)
    return _center_mask(shape, completed, color, _BG)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIZE_RANGE)
    w = random.randint(*_GRID_SIZE_RANGE)
    color_count = random.randint(*_COLOR_COUNT_RANGE)
    colors = random.sample(_COLORS, color_count)
    valid_color = random.choice(colors)

    valid_k = random.choice(_VALID_K_CHOICES)
    valid_corner = random.choice(_CORNERS)
    valid_fragment = _corner_l_mask(valid_k, valid_corner)

    placed: list[tuple[int, np.ndarray]] = [(valid_color, valid_fragment), (valid_color, valid_fragment)]
    catalog = _catalog()

    for color in colors:
        if color == valid_color:
            continue
        count = random.randint(*_COMPONENTS_PER_DISTRACTOR_COLOR)
        shapes = random.sample(catalog, count)
        for _k, _corner, mask in shapes:
            placed.append((color, mask.copy()))

    random.shuffle(placed)
    placed.sort(key=lambda item: int(np.sum(item[1])), reverse=True)

    grid = np.full((h, w), _BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for color, patch in placed:
        if not _place_patch(grid, blocked, patch, color):
            return None

    expected = _build_expected(grid.shape, valid_color, valid_fragment)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
