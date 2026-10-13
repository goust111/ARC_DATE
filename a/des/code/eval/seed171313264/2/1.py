from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRECTIONS = ("up", "right", "down", "left")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

GEN_BG = 0
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = DIR4 if connectivity == 4 else DIR8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0]
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _colorize(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    arr = np.full(mask.shape, int(bg), dtype=int)
    arr[mask] = int(color)
    return arr


def _tip_direction(mask: np.ndarray) -> str:
    mask = _tight_crop(mask.astype(bool))
    h, w = mask.shape
    full_sides = []
    if int(mask[0].sum()) == w:
        full_sides.append("up")
    if int(mask[-1].sum()) == w:
        full_sides.append("down")
    if int(mask[:, 0].sum()) == h:
        full_sides.append("left")
    if int(mask[:, -1].sum()) == h:
        full_sides.append("right")
    if len(full_sides) != 1:
        raise AssertionError(f"sprite should have exactly one flat base side, got {full_sides}")
    return OPPOSITE[full_sides[0]]


def _rotate_mask_to_direction(mask: np.ndarray, target_dir: str) -> np.ndarray:
    for k in range(4):
        rotated = _tight_crop(np.rot90(mask, k))
        try:
            if _tip_direction(rotated) == target_dir:
                return rotated
        except AssertionError:
            continue
    raise AssertionError(f"no rotation aligns sprite to {target_dir}")


def _find_fence_groups(grid: np.ndarray, bg: int) -> dict[int, list[dict[str, int]]]:
    h, w = grid.shape
    groups: dict[int, list[dict[str, int]]] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            fh = int(r1 - r0 + 1)
            fw = int(c1 - c0 + 1)
            if fh < 5 or fw < 5 or fh % 2 == 0 or fw % 2 == 0:
                continue
            if int(comp.shape[0]) != 2 * fh + 2 * fw - 5:
                continue
            sub = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            if bool(sub[1:-1, 1:-1].any()):
                continue
            border = np.zeros_like(sub, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            missing = np.argwhere(border & ~sub)
            if missing.shape[0] != 1:
                continue
            gr, gc = map(int, missing[0])
            gap_dir = None
            if gr == 0 and gc == fw // 2:
                gap_dir = "up"
            elif gr == fh - 1 and gc == fw // 2:
                gap_dir = "down"
            elif gc == 0 and gr == fh // 2:
                gap_dir = "left"
            elif gc == fw - 1 and gr == fh // 2:
                gap_dir = "right"
            if gap_dir is None:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.shape[0] < 1 or inner.shape[1] < 1:
                continue
            non_bg = np.argwhere(inner != bg)
            inner_center = (inner.shape[0] // 2, inner.shape[1] // 2)
            if non_bg.shape[0] != 1 or tuple(map(int, non_bg[0])) != inner_center:
                continue
            center_color = int(inner[inner_center])
            if center_color in (bg, color):
                continue
            groups.setdefault(color, []).append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "outer_h": int(fh),
                    "outer_w": int(fw),
                    "gap_dir": gap_dir,
                    "center_color": int(center_color),
                }
            )
    return groups


def _select_fences(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, int]]]:
    groups = _find_fence_groups(grid, bg)
    if not groups:
        raise AssertionError("no fence candidates found")
    ranked = []
    for color, fences in groups.items():
        total_area = sum(int(f["outer_h"]) * int(f["outer_w"]) for f in fences)
        ranked.append((len(fences), total_area, int(color), fences))
    ranked.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    _, _, fence_color, fences = ranked[0]
    return int(fence_color), fences


def _sprite_mask_for_color(grid: np.ndarray, color: int, forbidden: np.ndarray) -> np.ndarray:
    comps = _components((grid == color) & ~forbidden, connectivity=8)
    if not comps:
        raise AssertionError(f"no sprite found for color {color}")
    comp = max(comps, key=lambda coords: int(coords.shape[0]))
    r0, c0, r1, c1 = _bbox(comp)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
    return mask


def _stamp_centered(out: np.ndarray, mask: np.ndarray, color: int, fence: dict[str, int]) -> None:
    inner_h = int(fence["outer_h"]) - 2
    inner_w = int(fence["outer_w"]) - 2
    mh, mw = mask.shape
    if mh > inner_h or mw > inner_w:
        raise AssertionError("sprite does not fit inside fence")
    top = int(fence["r0"]) + 1 + (inner_h - mh) // 2
    left = int(fence["c0"]) + 1 + (inner_w - mw) // 2
    patch = out[top : top + mh, left : left + mw]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fence-center color to the outside monochrome sprite of the same color,
    rotate that sprite so its pointed side faces the fence gap, and stamp it centered
    inside the fence while preserving the rest of the grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    fence_color, fences = _select_fences(g, bg)
    forbidden = np.zeros_like(g, dtype=bool)
    for fence in fences:
        forbidden[int(fence["r0"]) : int(fence["r1"]) + 1, int(fence["c0"]) : int(fence["c1"]) + 1] = True

    out = g.copy()
    for fence in fences:
        color = int(fence["center_color"])
        sprite_mask = _sprite_mask_for_color(g, color, forbidden)
        rotated = _rotate_mask_to_direction(sprite_mask, str(fence["gap_dir"]))
        _stamp_centered(out, rotated, color, fence)
    return out


def _build_right_pointing_sprite(rng: random.Random, max_height: int, max_width: int) -> np.ndarray:
    height_choices = [value for value in (3, 5, 7) if value <= max_height]
    width_choices = [value for value in (3, 5) if value <= max_width]
    if not height_choices or not width_choices:
        raise AssertionError("sprite bounds too small")

    h = int(rng.choice(height_choices))
    w = int(rng.choice(width_choices))
    half = h // 2
    plateau = 1 if h >= 5 and rng.random() < 0.5 else 0
    min_width = 1 if rng.random() < 0.6 else 2
    widths = []
    for r in range(h):
        distance = max(0, abs(r - half) - plateau)
        widths.append(max(min_width, w - distance))
    if widths[0] >= w or widths[-1] >= w:
        widths[0] = max(1, w - 1)
        widths[-1] = max(1, w - 1)
    mask = np.zeros((h, w), dtype=bool)
    for r, width in enumerate(widths):
        mask[r, :width] = True
    if _tip_direction(mask) != "right":
        raise AssertionError("constructed sprite has wrong canonical direction")
    return mask


def _box_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(H, int(r0) + int(h) + pad)
    cc1 = min(W, int(c0) + int(w) + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = occupied.shape
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(H, int(r0) + int(h) + pad)
    cc1 = min(W, int(c0) + int(w) + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_fence(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    fence_color: int,
    gap_dir: str,
    center_color: int,
    bg: int,
) -> None:
    r1 = int(r0 + outer_h - 1)
    c1 = int(c0 + outer_w - 1)
    grid[r0, c0 : c1 + 1] = int(fence_color)
    grid[r1, c0 : c1 + 1] = int(fence_color)
    grid[r0 : r1 + 1, c0] = int(fence_color)
    grid[r0 : r1 + 1, c1] = int(fence_color)
    if gap_dir == "up":
        grid[r0, c0 + outer_w // 2] = int(bg)
    elif gap_dir == "down":
        grid[r1, c0 + outer_w // 2] = int(bg)
    elif gap_dir == "left":
        grid[r0 + outer_h // 2, c0] = int(bg)
    else:
        grid[r0 + outer_h // 2, c1] = int(bg)
    grid[r0 + outer_h // 2, c0 + outer_w // 2] = int(center_color)


def _draw_sprite(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _sample_specs(rng: random.Random) -> list[dict[str, object]]:
    count = int(rng.randint(2, 4))
    palette = list(range(1, 10))
    rng.shuffle(palette)
    fence_color = int(palette[0])
    item_colors = [int(color) for color in palette[1 : count + 1]]
    gap_dirs = list(DIRECTIONS)
    rng.shuffle(gap_dirs)

    specs: list[dict[str, object]] = []
    max_sprite_h = 7 if count <= 2 else 5
    max_sprite_w = 5 if count <= 3 else 3
    for idx in range(count):
        base_mask = _build_right_pointing_sprite(rng, max_sprite_h, max_sprite_w)
        start_dir = str(rng.choice(DIRECTIONS))
        input_mask = _rotate_mask_to_direction(base_mask, start_dir)
        max_dim = max(int(input_mask.shape[0]), int(input_mask.shape[1]))
        inner_h = max_dim + 2 * int(rng.choice((1, 2)))
        inner_w = max_dim + 2 * int(rng.choice((1, 2)))
        specs.append(
            {
                "fence_color": fence_color,
                "item_color": int(item_colors[idx]),
                "gap_dir": str(gap_dirs[idx]),
                "sprite_mask": input_mask,
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )
    return specs


def _place_specs(
    specs: list[dict[str, object]], H: int, W: int, rng: random.Random
) -> list[dict[str, object]] | None:
    occupied = np.zeros((H, W), dtype=bool)
    placed: list[dict[str, object] | None] = [None] * len(specs)

    fence_order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["outer_h"]) * int(specs[idx]["outer_w"]),
        reverse=True,
    )
    for idx in fence_order:
        spec = specs[idx]
        fh = int(spec["outer_h"])
        fw = int(spec["outer_w"])
        done = False
        for _ in range(200):
            r0 = int(rng.randint(1, H - fh - 1))
            c0 = int(rng.randint(1, W - fw - 1))
            if not _box_is_free(occupied, r0, c0, fh, fw, pad=1):
                continue
            placed[idx] = dict(spec)
            placed[idx]["fence_r0"] = int(r0)
            placed[idx]["fence_c0"] = int(c0)
            _mark_box(occupied, r0, c0, fh, fw, pad=1)
            done = True
            break
        if not done:
            return None

    edge_band = 7
    for idx, maybe_spec in enumerate(placed):
        assert maybe_spec is not None
        spec = maybe_spec
        sprite_mask = np.asarray(spec["sprite_mask"], dtype=bool)
        sh, sw = map(int, sprite_mask.shape)
        sides = list(DIRECTIONS)
        rng.shuffle(sides)
        done = False
        for side in sides:
            if side == "up":
                row_lo, row_hi = 0, edge_band - sh
                col_lo, col_hi = 0, W - sw
            elif side == "down":
                row_lo, row_hi = H - edge_band, H - sh
                col_lo, col_hi = 0, W - sw
            elif side == "left":
                row_lo, row_hi = 0, H - sh
                col_lo, col_hi = 0, edge_band - sw
            else:
                row_lo, row_hi = 0, H - sh
                col_lo, col_hi = W - edge_band, W - sw
            if row_hi < row_lo or col_hi < col_lo:
                continue
            for _ in range(120):
                r0 = int(rng.randint(row_lo, row_hi))
                c0 = int(rng.randint(col_lo, col_hi))
                if not _box_is_free(occupied, r0, c0, sh, sw, pad=1):
                    continue
                spec["sprite_r0"] = int(r0)
                spec["sprite_c0"] = int(c0)
                _mark_box(occupied, r0, c0, sh, sw, pad=1)
                done = True
                break
            if done:
                break
        if not done:
            return None

    return [dict(spec) for spec in placed if spec is not None]


def _render_input(placed_specs: Iterable[dict[str, object]], H: int, W: int) -> np.ndarray:
    grid = np.full((H, W), GEN_BG, dtype=int)
    placed_list = list(placed_specs)
    fence_color = int(placed_list[0]["fence_color"])
    for spec in placed_list:
        _draw_fence(
            grid,
            int(spec["fence_r0"]),
            int(spec["fence_c0"]),
            int(spec["outer_h"]),
            int(spec["outer_w"]),
            fence_color,
            str(spec["gap_dir"]),
            int(spec["item_color"]),
            GEN_BG,
        )
    for spec in placed_list:
        _draw_sprite(
            grid,
            np.asarray(spec["sprite_mask"], dtype=bool),
            int(spec["item_color"]),
            int(spec["sprite_r0"]),
            int(spec["sprite_c0"]),
        )
    return grid


def _expected_output(inp: np.ndarray, placed_specs: Iterable[dict[str, object]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in placed_specs:
        rotated = _rotate_mask_to_direction(np.asarray(spec["sprite_mask"], dtype=bool), str(spec["gap_dir"]))
        fence = {
            "r0": int(spec["fence_r0"]),
            "c0": int(spec["fence_c0"]),
            "outer_h": int(spec["outer_h"]),
            "outer_w": int(spec["outer_w"]),
        }
        _stamp_centered(out, rotated, int(spec["item_color"]), fence)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _sample_specs(rng)
    total_area = sum(
        int(spec["outer_h"]) * int(spec["outer_w"]) + int(np.asarray(spec["sprite_mask"]).sum())
        for spec in specs
    )
    base_side = max(22, int(np.ceil(np.sqrt(total_area * 2.6))))
    if base_side > 30:
        return None
    H = int(rng.randint(base_side, min(30, base_side + 4)))
    W = int(rng.randint(base_side, min(30, base_side + 4)))
    placed = _place_specs(specs, H, W, rng)
    if placed is None:
        return None
    inp = _render_input(placed, H, W)
    expected = _expected_output(inp, placed)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
