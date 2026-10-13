from __future__ import annotations

import random
from collections import Counter

import numpy as np


AXIS_COLOR = 2
OBSTACLE_COLOR = 5
CANVAS_SIZE = 30
GEN_RETRIES = 500
BG_CHOICES = (0, 1, 3, 4, 6, 7, 8, 9)
SPRITE_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _rects_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(
    grid: np.ndarray,
    *,
    colors: set[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    use_colors = None if colors is None else {int(c) for c in colors}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c]:
                continue
            if use_colors is not None and color not in use_colors:
                continue
            if use_colors is None and color == _get_bg(g):
                continue
            visited[r, c] = True
            q = [(r, c)]
            coords = [(r, c)]
            while q:
                rr, cc = q.pop()
                nbrs = _neighbors4(rr, cc) if connectivity == 4 else [
                    (rr + dr, cc + dc)
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                ]
                for nr, nc in nbrs:
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        coords.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return out


def _axis_info(component: dict) -> dict:
    coords = component["coords"]
    rs = sorted({r for r, _ in coords})
    cs = sorted({c for _, c in coords})
    if len(rs) == 1:
        row = rs[0]
        assert cs == list(range(cs[0], cs[-1] + 1)), "axis must be a contiguous horizontal line"
        return {"orientation": "h", "line": row, "span0": cs[0], "span1": cs[-1]}
    if len(cs) == 1:
        col = cs[0]
        assert rs == list(range(rs[0], rs[-1] + 1)), "axis must be a contiguous vertical line"
        return {"orientation": "v", "line": col, "span0": rs[0], "span1": rs[-1]}
    raise AssertionError("red component must be a straight axis segment")


def _pair_axis_to_sprite(axis: dict, sprites: list[dict], used: set[int]) -> tuple[int, str]:
    candidates: list[tuple[tuple[int, int, int, int], int, str]] = []
    for idx, sprite in enumerate(sprites):
        if idx in used:
            continue
        r0, c0, r1, c1 = sprite["bbox"]
        if axis["orientation"] == "h":
            if not (axis["span0"] <= c0 and c1 <= axis["span1"]):
                continue
            if r1 < axis["line"]:
                gap = axis["line"] - r1 - 1
                side = "above"
            elif r0 > axis["line"]:
                gap = r0 - axis["line"] - 1
                side = "below"
            else:
                continue
            score = (
                gap,
                (c0 - axis["span0"]) + (axis["span1"] - c1),
                abs((c0 + c1) - (axis["span0"] + axis["span1"])),
                sprite["area"],
            )
        else:
            if not (axis["span0"] <= r0 and r1 <= axis["span1"]):
                continue
            if c1 < axis["line"]:
                gap = axis["line"] - c1 - 1
                side = "left"
            elif c0 > axis["line"]:
                gap = c0 - axis["line"] - 1
                side = "right"
            else:
                continue
            score = (
                gap,
                (r0 - axis["span0"]) + (axis["span1"] - r1),
                abs((r0 + r1) - (axis["span0"] + axis["span1"])),
                sprite["area"],
            )
        if gap >= 1:
            candidates.append((score, idx, side))
    assert candidates, "each axis must have a visible paired sprite"
    candidates.sort(key=lambda item: item[0])
    assert len(candidates) == 1 or candidates[0][0] != candidates[1][0], "axis pairing must be unique"
    _, idx, side = candidates[0]
    return idx, side


def _mirror_coords(coords: list[tuple[int, int]], axis: dict) -> list[tuple[int, int]]:
    if axis["orientation"] == "h":
        row = axis["line"]
        return [(2 * row - r, c) for r, c in coords]
    col = axis["line"]
    return [(r, 2 * col - c) for r, c in coords]


def _slide_until_blocked(
    coords: list[tuple[int, int]],
    direction: tuple[int, int],
    occupied: np.ndarray,
) -> tuple[list[tuple[int, int]], int]:
    h, w = occupied.shape
    dr, dc = direction
    step = 0
    while True:
        shifted = [(r + (step + 1) * dr, c + (step + 1) * dc) for r, c in coords]
        blocked = False
        for nr, nc in shifted:
            if not (0 <= nr < h and 0 <= nc < w):
                blocked = True
                break
            if occupied[nr, nc]:
                blocked = True
                break
        if blocked:
            break
        step += 1
    return ([(r + step * dr, c + step * dc) for r, c in coords], step)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each sprite across its red axis and then eject the mirrored copy away from that axis until it hits a blocker.

    Input structure:
    - The grid has a dominant background plus several colored 4-connected sprites, several solid gray obstacle blocks,
      and several one-cell-thick red axis segments.
    - Every red segment is horizontal or vertical, lies strictly on one side of exactly one sprite, and spans past that
      sprite along the parallel direction without touching it.
    - Sprites, axes, and gray obstacles are pairwise disjoint, and each launch corridor has a definite stopper:
      either a gray obstacle ahead or the grid boundary.

    Rule:
    - For each sprite-axis pair, reflect the entire sprite across the red axis to the opposite side.
    - Move that mirrored copy step by step perpendicular to the axis, continuing away from the axis.
    - Stop just before the next step would overlap any existing input object or leave the grid, then draw the stopped
      mirrored copy. Keep all original sprites, all red axes, and all gray obstacles unchanged.

    Key insight:
    - The red segment determines both the reflection geometry and the launch direction; the reflected shape is only the
      starting pose, while the final landing point is fixed by the first obstacle or boundary encountered along that ray.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    axis_components = _find_components(g, colors={AXIS_COLOR}, connectivity=4)
    sprite_colors = set(int(v) for v in np.unique(g)) - {bg, AXIS_COLOR, OBSTACLE_COLOR}
    sprite_components = _find_components(g, colors=sprite_colors, connectivity=4)

    axes = [_axis_info(comp) for comp in axis_components]
    assert axes, "expected at least one red axis"
    assert sprite_components, "expected at least one sprite"

    occupied = g != bg
    out = g.copy()
    used_sprites: set[int] = set()

    for axis in axes:
        sprite_idx, side = _pair_axis_to_sprite(axis, sprite_components, used_sprites)
        used_sprites.add(sprite_idx)
        sprite = sprite_components[sprite_idx]
        mirrored = _mirror_coords(sprite["coords"], axis)
        for mr, mc in mirrored:
            assert 0 <= mr < h and 0 <= mc < w, "mirrored sprite must stay inside the grid"
            assert not occupied[mr, mc], "mirrored launch pose must start on empty cells"

        if axis["orientation"] == "h":
            direction = (1, 0) if side == "above" else (-1, 0)
        else:
            direction = (0, 1) if side == "left" else (0, -1)

        landed, slide = _slide_until_blocked(mirrored, direction, occupied)
        assert slide >= 1, "launch phase must move at least one step"
        color = int(sprite["color"])
        for rr, cc in landed:
            out[rr, cc] = color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return False
    start = coords[0]
    seen = {start}
    stack = [start]
    cells = set(coords)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        hh = rng.randint(2, 4)
        ww = rng.randint(2, 4)
        target = rng.randint(max(3, hh), min(hh * ww - 1, hh * ww))
        mask = np.zeros((hh, ww), dtype=bool)
        seed = (rng.randrange(hh), rng.randrange(ww))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r in range(hh):
                for c in range(ww):
                    if mask[r, c]:
                        continue
                    if any(0 <= nr < hh and 0 <= nc < ww and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = rng.choice(frontier)
            mask[rr, cc] = True
        if int(mask.sum()) != target:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not _mask_connected(mask):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric sprite mask")


def _canonical_group(
    rng: random.Random, color: int, blocker: str, bg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    mask = _random_sprite_mask(rng)
    hh, ww = mask.shape
    gap = rng.randint(1, 3)
    slide = rng.randint(1, 4)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    axis_len = left_margin + ww + right_margin
    axis_row = hh + gap
    mirror_row = axis_row + gap + 1
    final_row = mirror_row + slide

    obstacle_h = 0
    if blocker == "obstacle":
        obstacle_h = rng.randint(1, 2)
    patch_h = final_row + hh + obstacle_h
    patch_w = axis_len

    inp = np.full((patch_h, patch_w), bg, dtype=int)
    out = np.full((patch_h, patch_w), bg, dtype=int)
    mirror_only = np.full((patch_h, patch_w), bg, dtype=int)
    no_mirror = np.full((patch_h, patch_w), bg, dtype=int)

    sprite_cols = list(range(left_margin, left_margin + ww))
    sprite_cells = [(r, c) for r in range(hh) for c in range(ww) if mask[r, c]]
    reflected_cells = [(hh - 1 - r, c) for r, c in sprite_cells]

    for canvas in (inp, out, mirror_only, no_mirror):
        for c in range(axis_len):
            canvas[axis_row, c] = AXIS_COLOR
        for r, c in sprite_cells:
            canvas[r, left_margin + c] = color

    if blocker == "obstacle":
        obstacle_row = final_row + hh
        inp[obstacle_row : obstacle_row + obstacle_h, :] = OBSTACLE_COLOR
        out[obstacle_row : obstacle_row + obstacle_h, :] = OBSTACLE_COLOR
        mirror_only[obstacle_row : obstacle_row + obstacle_h, :] = OBSTACLE_COLOR
        no_mirror[obstacle_row : obstacle_row + obstacle_h, :] = OBSTACLE_COLOR

    for r, c in reflected_cells:
        mirror_only[mirror_row + r, left_margin + c] = color
        out[final_row + r, left_margin + c] = color
    for r, c in sprite_cells:
        no_mirror[final_row + r, left_margin + c] = color

    meta = {
        "blocker": blocker,
        "slide": slide,
        "orientation": "h",
        "side": "above",
        "axis_gap": gap,
    }
    return inp, out, mirror_only, no_mirror, meta


def _apply_mode(arr: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return arr.copy()
    if mode == 1:
        return np.flipud(arr).copy()
    if mode == 2:
        return arr.T.copy()
    if mode == 3:
        return np.fliplr(arr.T).copy()
    raise ValueError(f"bad mode {mode}")


def _mode_meta(meta: dict, mode: int) -> tuple[str, str, str | None]:
    if mode == 0:
        return "h", "above", "bottom"
    if mode == 1:
        return "h", "below", "top"
    if mode == 2:
        return "v", "left", "right"
    if mode == 3:
        return "v", "right", "left"
    raise ValueError(f"bad mode {mode}")


def _place_patches(
    rng: random.Random,
    patches: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_in = np.full((CANVAS_SIZE, CANVAS_SIZE), patches[0]["bg"], dtype=int)
    grid_out = np.full_like(grid_in, patches[0]["bg"])
    grid_mirror = np.full_like(grid_in, patches[0]["bg"])
    grid_no_mirror = np.full_like(grid_in, patches[0]["bg"])
    placed_rects: list[tuple[int, int, int, int]] = []

    ordered = sorted(
        range(len(patches)),
        key=lambda i: (
            0 if patches[i]["boundary_side"] is not None else 1,
            -(patches[i]["inp"].shape[0] * patches[i]["inp"].shape[1]),
        ),
    )

    for idx in ordered:
        patch = patches[idx]
        hh, ww = patch["inp"].shape
        side = patch["boundary_side"]
        candidates: list[tuple[int, int]] = []
        if side == "top":
            candidates = [(0, c) for c in range(CANVAS_SIZE - ww + 1)]
        elif side == "bottom":
            candidates = [(CANVAS_SIZE - hh, c) for c in range(CANVAS_SIZE - ww + 1)]
        elif side == "left":
            candidates = [(r, 0) for r in range(CANVAS_SIZE - hh + 1)]
        elif side == "right":
            candidates = [(r, CANVAS_SIZE - ww) for r in range(CANVAS_SIZE - hh + 1)]
        else:
            candidates = [
                (r, c)
                for r in range(CANVAS_SIZE - hh + 1)
                for c in range(CANVAS_SIZE - ww + 1)
            ]
        rng.shuffle(candidates)

        placed = False
        for top, left in candidates:
            rect = (top, left, top + hh - 1, left + ww - 1)
            if all(_rects_separated(rect, other, gap=1) for other in placed_rects):
                _paste(grid_in, patch["inp"], top, left, transparent=patch["bg"])
                _paste(grid_out, patch["out"], top, left, transparent=patch["bg"])
                _paste(grid_mirror, patch["mirror"], top, left, transparent=patch["bg"])
                _paste(grid_no_mirror, patch["no_mirror"], top, left, transparent=patch["bg"])
                placed_rects.append(rect)
                patch["rect"] = rect
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place patch")

    constrained = {
        "top": any(p["boundary_side"] == "top" for p in patches),
        "bottom": any(p["boundary_side"] == "bottom" for p in patches),
        "left": any(p["boundary_side"] == "left" for p in patches),
        "right": any(p["boundary_side"] == "right" for p in patches),
    }
    union_r0 = min(rect[0] for rect in placed_rects)
    union_c0 = min(rect[1] for rect in placed_rects)
    union_r1 = max(rect[2] for rect in placed_rects)
    union_c1 = max(rect[3] for rect in placed_rects)
    crop_r0 = 0 if constrained["top"] else rng.randint(0, union_r0)
    crop_c0 = 0 if constrained["left"] else rng.randint(0, union_c0)
    crop_r1 = CANVAS_SIZE - 1 if constrained["bottom"] else rng.randint(union_r1, CANVAS_SIZE - 1)
    crop_c1 = CANVAS_SIZE - 1 if constrained["right"] else rng.randint(union_c1, CANVAS_SIZE - 1)

    sl = (slice(crop_r0, crop_r1 + 1), slice(crop_c0, crop_c1 + 1))
    return grid_in[sl].copy(), grid_out[sl].copy(), grid_mirror[sl].copy(), grid_no_mirror[sl].copy()


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    bg = rng.choice(BG_CHOICES)
    available_colors = [c for c in SPRITE_COLOR_POOL if c not in {bg, AXIS_COLOR, OBSTACLE_COLOR}]
    group_count = rng.randint(2, min(4, len(available_colors)))
    colors = rng.sample(available_colors, group_count)

    blockers = ["obstacle", "boundary"]
    while len(blockers) < group_count:
        blockers.append(rng.choice(["obstacle", "boundary"]))
    rng.shuffle(blockers)

    modes = [rng.choice([0, 1]), rng.choice([2, 3])]
    while len(modes) < group_count:
        modes.append(rng.randrange(4))
    rng.shuffle(modes)

    patches: list[dict] = []
    metas: list[dict] = []
    for color, blocker, mode in zip(colors, blockers, modes):
        inp, out, mirror_only, no_mirror, meta = _canonical_group(rng, color, blocker, bg)
        inp = _apply_mode(inp, mode)
        out = _apply_mode(out, mode)
        mirror_only = _apply_mode(mirror_only, mode)
        no_mirror = _apply_mode(no_mirror, mode)
        orientation, side, boundary_side = _mode_meta(meta, mode)
        if blocker != "boundary":
            boundary_side = None
        metas.append(
            {
                "color": color,
                "blocker": blocker,
                "slide": meta["slide"],
                "orientation": orientation,
                "side": side,
            }
        )
        patches.append(
            {
                "inp": inp,
                "out": out,
                "mirror": mirror_only,
                "no_mirror": no_mirror,
                "boundary_side": boundary_side,
                "bg": bg,
            }
        )

    inp, out, mirror_only, no_mirror = _place_patches(rng, patches)
    return inp, out, mirror_only, no_mirror, metas


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    mirror_only: np.ndarray,
    no_mirror: np.ndarray,
    metas: list[dict],
) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, mirror_only):
        return False
    if np.array_equal(out, no_mirror):
        return False
    if not all(meta["slide"] >= 1 for meta in metas):
        return False
    if not any(meta["slide"] >= 2 for meta in metas):
        return False
    if len(metas) >= 2:
        if not any(meta["blocker"] == "boundary" for meta in metas):
            return False
        if not any(meta["blocker"] == "obstacle" for meta in metas):
            return False
        if not any(meta["orientation"] == "h" for meta in metas):
            return False
        if not any(meta["orientation"] == "v" for meta in metas):
            return False
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, out):
        return False
    bg = _get_bg(inp)
    if int(np.count_nonzero(inp != bg)) >= inp.size // 2:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            inp, out, mirror_only, no_mirror, metas = _build_case(rng)
        except Exception:
            continue
        if _validate_semantics(inp, out, mirror_only, no_mirror, metas):
            return inp
    raise RuntimeError("failed to generate a valid mirror-ejection input")
