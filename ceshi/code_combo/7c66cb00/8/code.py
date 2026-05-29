# Combo variant of 7c66cb00 (index 8)
# combo_spec: D3/0+D4/2+D6/0
# Name: 空心框架接收三色精灵保持颜色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization + behavior trace
#    - Original task:
#      a 2-color sprite above the bars chooses a destination by matching its outer color to a
#      bar fill, then the sprite drops into that bar and its colors are swapped to bar colors.
#    - This combo:
#      the destinations are separated hollow rectangular frames, not full-width bars, and the
#      source object is a 3-color nested sprite (outer shell, middle band, core). Example:
#        input top sprite = 5x7 patch with colors [A outside, B in a centered 3x5 band,
#        C in a centered 1x3 core]. Lower area has one hollow frame whose border color is A.
#        output = erase the top sprite, then paste the same 5x7 tri-color patch unchanged into
#        that frame's 5x7 interior. No splitting, no recolor, no reuse of the old x-position.
#    - Difference from the original:
#      matching moved from bar-fill matching to frame-border matching; placement moved from
#      1D bottom docking inside a bar to exact 2D insertion into a frame cavity; the 3-color
#      structure is preserved as a whole instead of being recolored or decomposed.
# b) Complexity calibration
#    - We keep exactly 3 hollow frames so every sample supports a genuine 3-color palette and
#      every sprite can use all 3 frame-border colors without introducing extra arbitrary colors.
#    - All frame interiors share the same 5x7 size. This intentionally removes size as a unique
#      destination cue, forcing border-color matching to do real work while still keeping the
#      "fills the cavity exactly" property visually obvious.
#    - There are 2-3 sprites, each targeting a different frame by outer color. This avoids
#      ambiguous multi-sprite packing inside one frame and makes the preserved-color rule legible.
# c) Hidden assumption audit against the original main()
#    - Original row-scan bar detector assumed wide filled rows. Hollow local rectangles violate
#      that, so frame detection must switch to connected-component bbox validation.
#    - Original sprite extraction iterated per color component. With 3 nested colors that would
#      rediscover one visual sprite multiple times. The combo must detect whole non-bg components.
#    - Original solver assumed at most one inner color and a 2-color recolor loop. The combo
#      needs explicit 3-layer parsing, but the output patch is copied unchanged.
#    - Original placement reused the sprite's input x-coordinate because bars span the canvas.
#      Here that would be meaningless: the frame interior defines the destination coordinates.
# d) Placement legality
#    - Generator places frames with at least a 1-cell bg moat so each frame is a separate
#      connected component and the hollow border is visually isolated.
#    - Top sprites are also separated by at least 1 bg cell, and we additionally keep at least
#      1 full bg row between the top sprite zone and the highest frame so sprites never merge
#      with frames during component detection.
#    - Every sprite's size is forced to equal its target frame's interior size exactly, so the
#      output placement is always in-bounds and fills the frame cavity without overlap.
# e) Visual observability
#    - The lower objects visibly read as empty boxes before transformation and partially filled
#      boxes after transformation. The human clue is "the colored nested patch moves into the
#      box with the same outer-ring color on its border".
#    - Because the pasted sprite keeps all 3 colors, examples show that inner and core colors do
#      not trigger extra destinations; only the outer ring chooses the frame.
# f) Visual cleanliness
#    - The output contains no overdraw inside a frame because each frame receives at most one
#      sprite and the sprite exactly matches the frame interior dimensions.
#    - Using solid nested rectangles avoids noisy silhouettes and keeps the preserved-color rule
#      easy to compare between input and output.
# g) Change inferability
#    - There is no hidden color remapping: every output sprite uses the identical 3-color set and
#      identical local geometry seen in the input. The only change is relocation into a matching
#      hollow frame, which is structurally visible from a few examples.
# 9) New solver assumptions and generator guarantees
#    - main() assumes each top sprite is one connected component with exactly 3 colors and that
#      exactly one of those colors touches the sprite bbox border. Generation enforces this with
#      nested filled rectangles.
#    - main() assumes each hollow frame is a single-color border rectangle with an all-bg
#      interior. Generation builds frames in exactly that form.
#    - main() assumes the sprite matched by a frame border fits that frame interior exactly.
#      Generation sets each sprite width/height from its matched frame interior dimensions.
#    - main() assumes no two sprites target the same frame. Generation assigns distinct outer
#      colors, so each border color is used by at most one sprite.


GRID_H = 30
GRID_W = 24

FRAME_COUNT = 3
SPRITE_COUNT_RANGE = (2, 3)
FRAME_INNER_H = 5
FRAME_INNER_W = 7
MIN_TOP_AREA = 8
MIN_FRAME_SPREAD_R = 3
MIN_FRAME_SPREAD_C = 4

GEN_RETRIES = 500
PLACEMENT_RETRIES = 300


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) <= int(inner[0])
        and int(outer[1]) <= int(inner[1])
        and int(outer[2]) >= int(inner[2])
        and int(outer[3]) >= int(inner[3])
    )


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _extract_layers(patch: np.ndarray, bg: int) -> dict[str, dict]:
    infos = []
    h, w = map(int, patch.shape)
    for color in sorted(set(map(int, patch.flatten())) - {int(bg)}):
        mask = patch == int(color)
        coords = np.argwhere(mask)
        bbox = tuple(map(int, bbox_coords(coords)))
        touches_border = bool(
            mask[0, :].any() or mask[h - 1, :].any() or mask[:, 0].any() or mask[:, w - 1].any()
        )
        infos.append(
            {
                "color": int(color),
                "bbox": bbox,
                "bbox_area": _bbox_area(bbox),
                "touches_border": touches_border,
            }
        )

    assert len(infos) == 3, "Each sprite must contain exactly three colors"

    outer_candidates = [info for info in infos if info["touches_border"]]
    assert len(outer_candidates) == 1, "Expected exactly one outer layer touching the bbox edge"
    outer = outer_candidates[0]

    nested = [info for info in infos if info["color"] != int(outer["color"])]
    nested.sort(key=lambda info: int(info["bbox_area"]), reverse=True)
    middle, core = nested

    assert _contains(tuple(outer["bbox"]), tuple(middle["bbox"]))
    assert _contains(tuple(middle["bbox"]), tuple(core["bbox"]))
    return {"outer": outer, "middle": middle, "core": core}


def _frame_from_region(region: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict | None:
    h, w = map(int, region.shape)
    if h < 5 or w < 5:
        return None

    border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
    border_color, border_count = Counter(map(int, border.flatten())).most_common(1)[0]
    if int(border_color) == int(bg) or int(border_count) != int(border.size):
        return None

    interior = region[1:-1, 1:-1]
    if interior.size == 0 or not np.all(interior == int(bg)):
        return None

    r0, c0, r1, c1 = map(int, bbox)
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(h),
        "w": int(w),
        "border": int(border_color),
        "inner_h": int(h - 2),
        "inner_w": int(w - 2),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(grid != int(bg), connectivity=4)
    frames = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        bbox = tuple(map(int, bbox_coords(coords)))
        frame = _frame_from_region(crop_bbox(grid, bbox), bbox, bg)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda frame: (int(frame["r0"]), int(frame["c0"])))
    return frames


def _find_sprites(grid: np.ndarray, bg: int, top_end: int) -> list[dict]:
    top = grid[:top_end, :]
    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = crop_bbox(top, (r0, c0, r1, c1))
        layers = _extract_layers(content, bg)
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "content": np.array(content, dtype=int),
                "layers": layers,
                "outer_color": int(layers["outer"]["color"]),
            }
        )
    sprites.sort(key=lambda sprite: (int(sprite["r0"]), int(sprite["c0"])))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move three-color sprites into border-matched hollow frames without changing colors.

    Input structure:
    - Background is the most common color.
    - Top area contains separated 3-color nested sprites: an outer ring, a middle band,
      and a centered core, all as one connected component.
    - Lower area contains separated hollow rectangular frames whose interiors are background.

    Rule:
    - Detect each whole sprite as one component and identify its outer color.
    - Match the sprite to the hollow frame whose border color equals that outer color.
    - Erase the sprite from the top area.
    - Paste the unchanged sprite into the matched frame so it exactly fills the frame interior.

    Key insight: the destination cue is now the frame border itself, and the tri-color sprite
    is transferred intact as a single object rather than recolored or split across targets.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames = _find_frames(g, bg)
    assert len(frames) == FRAME_COUNT, "Expected exactly three hollow frames"
    frames_by_border = {int(frame["border"]): frame for frame in frames}
    assert len(frames_by_border) == len(frames), "Frame border colors must be unique"

    top_end = int(min(int(frame["r0"]) for frame in frames))
    sprites = _find_sprites(g, bg, top_end)

    out[:top_end, :] = np.where(out[:top_end, :] != int(bg), int(bg), out[:top_end, :])

    for sprite in sprites:
        frame = frames_by_border[int(sprite["outer_color"])]
        patch = np.array(sprite["content"], dtype=int)
        assert int(patch.shape[0]) == int(frame["inner_h"])
        assert int(patch.shape[1]) == int(frame["inner_w"])
        paste(out, patch, int(frame["r0"] + 1), int(frame["c0"] + 1), transparent=int(bg))

    return out


def _place_frames(frame_dims: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        order = list(range(len(frame_dims)))
        random.shuffle(order)
        rects: list[tuple[int, int, int, int]] = []
        placed: dict[int, tuple[int, int]] = {}

        for idx in order:
            h, w = map(int, frame_dims[idx])
            candidates = []
            for r0 in range(MIN_TOP_AREA, GRID_H - h + 1):
                for c0 in range(0, GRID_W - w + 1):
                    if _rect_separated(rects, r0, c0, h, w):
                        candidates.append((int(r0), int(c0)))
            if not candidates:
                break
            choice = random.choice(candidates)
            placed[idx] = choice
            rects.append((int(choice[0]), int(choice[1]), int(choice[0] + h - 1), int(choice[1] + w - 1)))

        if len(placed) != len(frame_dims):
            continue

        positions = [placed[i] for i in range(len(frame_dims))]
        rows = [int(pos[0]) for pos in positions]
        cols = [int(pos[1]) for pos in positions]
        if max(rows) - min(rows) < MIN_FRAME_SPREAD_R:
            continue
        if max(cols) - min(cols) < MIN_FRAME_SPREAD_C:
            continue
        return positions
    return None


def _build_frames(bg: int, border_colors: list[int]) -> list[dict] | None:
    inner_widths = [FRAME_INNER_W] * FRAME_COUNT
    frame_dims = [(FRAME_INNER_H + 2, FRAME_INNER_W + 2) for _ in range(FRAME_COUNT)]
    positions = _place_frames(frame_dims)
    if positions is None:
        return None

    frames = []
    for idx, (border, inner_w, dims, pos) in enumerate(zip(border_colors, inner_widths, frame_dims, positions)):
        h, w = map(int, dims)
        r0, c0 = map(int, pos)
        frames.append(
            {
                "idx": int(idx),
                "border": int(border),
                "h": int(h),
                "w": int(w),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + h - 1),
                "c1": int(c0 + w - 1),
                "inner_h": int(FRAME_INNER_H),
                "inner_w": int(inner_w),
            }
        )
    return frames


def _build_base(bg: int, frames: list[dict]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for frame in frames:
        r0, c0 = int(frame["r0"]), int(frame["c0"])
        h, w = int(frame["h"]), int(frame["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(frame["border"])
        grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = int(bg)
    return grid


def _make_sprite(order: tuple[int, int, int], frames: list[dict]) -> dict:
    outer_idx, middle_idx, core_idx = map(int, order)
    w = int(frames[outer_idx]["inner_w"])
    patch = np.full((FRAME_INNER_H, w), int(frames[outer_idx]["border"]), dtype=int)

    middle_width_choices = [mw for mw in (w - 2, w - 4) if mw >= 3]
    middle_w = int(random.choice(middle_width_choices))
    core_width_choices = [cw for cw in range(1, middle_w, 2) if cw <= middle_w - 2]
    core_w = int(random.choice(core_width_choices))

    middle_bbox = (1, (w - middle_w) // 2, 3, middle_w)
    core_bbox = (2, (w - core_w) // 2, 1, core_w)

    mr0, mc0, mh, mw = map(int, middle_bbox)
    cr0, cc0, ch, cw = map(int, core_bbox)
    patch[mr0 : mr0 + mh, mc0 : mc0 + mw] = int(frames[middle_idx]["border"])
    patch[cr0 : cr0 + ch, cc0 : cc0 + cw] = int(frames[core_idx]["border"])

    return {
        "outer_idx": int(outer_idx),
        "middle_idx": int(middle_idx),
        "core_idx": int(core_idx),
        "patch": patch,
        "h": int(FRAME_INNER_H),
        "w": int(w),
        "target_r0": int(frames[outer_idx]["r0"] + 1),
        "target_c0": int(frames[outer_idx]["c0"] + 1),
    }


def _make_sprites(frames: list[dict], n_sprites: int) -> list[dict]:
    target_indices = list(range(FRAME_COUNT))
    random.shuffle(target_indices)
    target_indices = target_indices[:n_sprites]

    sprites = []
    all_indices = list(range(FRAME_COUNT))
    for outer_idx in target_indices:
        remaining = [idx for idx in all_indices if idx != outer_idx]
        random.shuffle(remaining)
        sprites.append(_make_sprite((outer_idx, remaining[0], remaining[1]), frames))
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), int(sprite["outer_idx"])))
    return sprites


def _place_top_sprites(sprites: list[dict], top_end: int) -> list[dict] | None:
    rects: list[tuple[int, int, int, int]] = []
    placed = []
    max_bottom = int(top_end - 2)

    for sprite in sprites:
        h, w = int(sprite["h"]), int(sprite["w"])
        if h - 1 > max_bottom:
            return None

        candidates = []
        for r0 in range(0, max_bottom - h + 2):
            for c0 in range(0, GRID_W - w + 1):
                if _rect_separated(rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, sprites: list[dict], frames: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for sprite in sprites:
        patch = np.array(sprite["patch"], dtype=int)
        paste(inp, patch, int(sprite["r0"]), int(sprite["c0"]), transparent=int(bg))
        paste(out, patch, int(sprite["target_r0"]), int(sprite["target_c0"]), transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    border_colors = [int(v) for v in palette[1 : 1 + FRAME_COUNT]]

    frames = _build_frames(bg, border_colors)
    if frames is None:
        return None

    top_end = min(int(frame["r0"]) for frame in frames)
    if top_end < MIN_TOP_AREA:
        return None

    sprites = _make_sprites(frames, n_sprites)
    placed = _place_top_sprites(sprites, top_end)
    if placed is None:
        return None

    base = _build_base(bg, frames)
    inp, out = _render(base, placed, frames, bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
