from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(int(x) for x in mask.ravel())


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                    "mask": _component_mask(arr),
                }
            )
    return out


def _equal_step(values: list[int]) -> bool:
    if len(values) <= 1:
        return True
    step = values[1] - values[0]
    return step > 0 and all(values[i + 1] - values[i] == step for i in range(len(values) - 1))


def _layout_mask(components: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    centers = [(int(c["bbox"][0]) + int(c["bbox"][2]), int(c["bbox"][1]) + int(c["bbox"][3])) for c in components]
    row_vals = sorted({r for r, _ in centers})
    col_vals = sorted({c for _, c in centers})
    if len(row_vals) != int(shape[0]) or len(col_vals) != int(shape[1]):
        return None
    if not _equal_step(row_vals) or not _equal_step(col_vals):
        return None

    row_index = {v: i for i, v in enumerate(row_vals)}
    col_index = {v: i for i, v in enumerate(col_vals)}
    layout = np.zeros((len(row_vals), len(col_vals)), dtype=bool)
    for r, c in centers:
        rr, cc = row_index[r], col_index[c]
        if layout[rr, cc]:
            return None
        layout[rr, cc] = True
    return layout


def _class_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    groups: dict[tuple[int, tuple[tuple[int, int], tuple[int, ...]]], list[dict]] = defaultdict(list)
    masks: dict[tuple[int, tuple[tuple[int, int], tuple[int, ...]]], np.ndarray] = {}

    for comp in _find_components(g, bg):
        mask = comp["mask"]
        key = (int(comp["color"]), _mask_key(mask))
        groups[key].append(comp)
        masks[key] = mask

    infos: list[dict] = []
    for key, comps in groups.items():
        color = key[0]
        mask = masks[key]
        count = len(comps)
        area = int(mask.sum())
        layout = _layout_mask(comps, mask.shape) if count == area and count > 1 else None
        self_similar = layout is not None and bool(np.array_equal(layout, mask))
        infos.append(
            {
                "key": key,
                "color": color,
                "mask": mask,
                "components": comps,
                "count": count,
                "area": area,
                "layout": layout,
                "self_similar": self_similar,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the sprite whose copy layout is the same shape as the sprite itself.

    Input structure:
    - A dominant background contains several classes of small monochrome sprites.
    - A class is defined by one color and one exact bounding-box shape; every copy
      is unrotated and separated from the other copies.
    - Exactly one class has copy-center positions lying on an evenly spaced row
      and column lattice whose occupied cells reproduce that class's sprite mask.

    Rule:
    - Group all monochrome components by color and normalized bounding-box mask.
    - For each group, use the bounding-box centers of all copies to build the
      occupied positions of the equal-spaced center lattice.
    - Select the unique group whose lattice occupancy has the same dimensions
      and exact mask as its single sprite, then output one cropped sprite.

    Key insight:
    - The decisive evidence is cross-scale identity: the small internal sprite
      shape and the large arrangement of its copies must be the same pattern.
      Copy count alone, largest area, and highest frequency are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"
    bg = _get_bg(g)
    winners = [info for info in _class_infos(g) if info["self_similar"]]
    assert len(winners) == 1, "expected exactly one self-similar sprite class"

    winner = winners[0]
    out = np.full(winner["mask"].shape, bg, dtype=int)
    out[winner["mask"]] = int(winner["color"])
    return out


# ----------------------------- generator helpers -----------------------------


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _valid_masks(height: int, width: int, area: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    total = height * width
    for cells in combinations(range(total), area):
        mask = np.zeros((height, width), dtype=bool)
        for k in cells:
            mask[k // width, k % width] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _connected(mask):
            masks.append(mask)
    return masks


_ANSWER_SPECS_RAW = ((2, 3, 4), (3, 2, 4), (3, 3, 5), (3, 3, 6))
ANSWER_MASKS_BY_SPEC = {spec: _valid_masks(*spec) for spec in _ANSWER_SPECS_RAW}
ANSWER_SPECS = tuple(spec for spec, masks in ANSWER_MASKS_BY_SPEC.items() if len(masks) >= 2)
LARGE_MASKS = {area: _valid_masks(4, 4, area) for area in (7, 8, 9)}
HIGH_MASKS = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
)


def _sprite(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _render_lattice_class(sprite_mask: np.ndarray, layout_mask: np.ndarray, color: int, pitch: int) -> np.ndarray:
    sh, sw = sprite_mask.shape
    lh, lw = layout_mask.shape
    patch = np.full((sh + (lh - 1) * pitch, sw + (lw - 1) * pitch), BG, dtype=int)
    sprite = _sprite(sprite_mask, color)
    for lr, lc in np.argwhere(layout_mask):
        r = int(lr) * pitch
        c = int(lc) * pitch
        patch[r : r + sh, c : c + sw][sprite_mask] = sprite[sprite_mask]
    return patch


def _patch_fg(patch: np.ndarray) -> np.ndarray:
    return np.argwhere(patch != BG)


def _mark_foreground_padding(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _mark_rect_padding(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, int(top) - 1)
    c0 = max(0, int(left) - 1)
    r1 = min(blocked.shape[0], int(top) + int(height) + 1)
    c1 = min(blocked.shape[1], int(left) + int(width) + 1)
    blocked[r0:r1, c0:c1] = True


def _can_place(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, top: int, left: int, reserve_rect: bool) -> bool:
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > grid.shape[0] or left + pw > grid.shape[1]:
        return False
    if reserve_rect:
        return not bool(blocked[top : top + ph, left : left + pw].any())

    fg = _patch_fg(patch)
    if len(fg) == 0:
        return False
    rows = top + fg[:, 0]
    cols = left + fg[:, 1]
    return not bool(blocked[rows, cols].any()) and bool(np.all(grid[rows, cols] == BG))


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, *, reserve_rect: bool) -> bool:
    ph, pw = patch.shape
    positions = [(r, c) for r in range(grid.shape[0] - ph + 1) for c in range(grid.shape[1] - pw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place(grid, blocked, patch, top, left, reserve_rect):
            continue

        fg = _patch_fg(patch)
        rows = top + fg[:, 0]
        cols = left + fg[:, 1]
        grid[rows, cols] = patch[fg[:, 0], fg[:, 1]]
        if reserve_rect:
            _mark_rect_padding(blocked, top, left, ph, pw)
        else:
            _mark_foreground_padding(blocked, np.column_stack([rows, cols]))
        return True
    return False


def _place_many(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int, count: int) -> bool:
    patch = _sprite(mask, color)
    for _ in range(int(count)):
        if not _place_patch(grid, blocked, patch, reserve_rect=False):
            return False
    return True


def _different_same_area_mask(answer: np.ndarray) -> np.ndarray:
    area = int(answer.sum())
    spec = (int(answer.shape[0]), int(answer.shape[1]), area)
    options = [
        m
        for m in ANSWER_MASKS_BY_SPEC[spec]
        if not np.array_equal(m, answer) and int(np.count_nonzero(m != answer)) >= 2
    ]
    return random.choice(options).copy()


def _make_expected(mask: np.ndarray, color: int) -> np.ndarray:
    return _sprite(mask, color)


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, answer_color: int, answer_mask: np.ndarray) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    infos = _class_infos(grid)
    answer_key = (int(answer_color), _mask_key(answer_mask))
    self_similar = [info for info in infos if info["self_similar"]]
    if len(self_similar) != 1 or self_similar[0]["key"] != answer_key:
        return False

    max_count = max(info["count"] for info in infos)
    if any(info["key"] == answer_key and info["count"] == max_count for info in infos):
        return False

    max_area = max(info["area"] for info in infos)
    if any(info["key"] == answer_key and info["area"] == max_area for info in infos):
        return False

    equal_area_false = [
        info
        for info in infos
        if info["key"] != answer_key
        and info["count"] == info["area"]
        and not info["self_similar"]
        and info["layout"] is not None
    ]
    if not equal_area_false:
        return False

    answer_layout_decoy = any(info["layout"] is not None and np.array_equal(info["layout"], answer_mask) for info in equal_area_false)
    return bool(answer_layout_decoy)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(27, 30)
    width = random.randint(27, 30)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    answer_spec = random.choice(ANSWER_SPECS)
    answer_area = answer_spec[2]
    answer_mask = random.choice(ANSWER_MASKS_BY_SPEC[answer_spec]).copy()
    fake_mask = _different_same_area_mask(answer_mask)
    high_mask = random.choice(HIGH_MASKS).copy()
    large_area = random.randint(max(answer_area + 2, 7), 9)
    large_mask = random.choice(LARGE_MASKS[large_area]).copy()

    answer_color, fake_color, high_color, large_color, extra_color = random.sample(COLORS, 5)
    pitch = 5
    answer_patch = _render_lattice_class(answer_mask, answer_mask, answer_color, pitch)
    fake_patch = _render_lattice_class(fake_mask, answer_mask, fake_color, pitch)

    clusters = [answer_patch, fake_patch]
    random.shuffle(clusters)
    for patch in clusters:
        if not _place_patch(grid, blocked, patch, reserve_rect=True):
            return None

    largest_count = random.choice((2, 3))
    if largest_count == large_area:
        largest_count = 2
    if not _place_many(grid, blocked, large_mask, large_color, largest_count):
        return None

    high_count = answer_area + random.randint(2, 3)
    if high_count == int(high_mask.sum()):
        high_count += 1
    if not _place_many(grid, blocked, high_mask, high_color, high_count):
        return None

    if random.random() < 0.55:
        extra_mask = random.choice(HIGH_MASKS).copy()
        extra_area = int(extra_mask.sum())
        extra_count = random.choice((2, 3, 4))
        if extra_count == extra_area:
            extra_count = 2 if extra_area != 2 else 3
        if extra_count < high_count and extra_area < large_area:
            if not _place_many(grid, blocked, extra_mask, extra_color, extra_count):
                return None

    expected = _make_expected(answer_mask, answer_color)
    if _semantic_checks(grid, expected, answer_color, answer_mask):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-similar layout input")
