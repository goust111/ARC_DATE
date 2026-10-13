from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
MIRROR = Color.GREY
WINDOW = Color.TEAL
BAR_COLORS = (MIRROR, WINDOW)
SPRITE_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]

MAX_GROUP_ATTEMPTS = 300
MAX_CASE_ATTEMPTS = 300


def _bar_key(bar: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, bar["bbox"])
    return (r0, c0, r1, c1, int(bar["color"]))


def _clear_sprites(grid: np.ndarray, sprite_grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[sprite_grid != bg] = bg
    return out


def _find_bar_components(grid: np.ndarray, bg: int) -> list[dict]:
    bar_grid = np.where(np.isin(grid, BAR_COLORS), grid, bg)
    bars = find_components(
        bar_grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=BAR_COLORS,
        return_format="dict",
    )
    parsed = []
    for bar in bars:
        r0, c0, r1, c1 = map(int, bar["bbox"])
        assert c0 == c1, "mirror bars must be one column wide"
        assert int(bar["area"]) == r1 - r0 + 1, "mirror bars must be solid vertical segments"
        parsed.append(bar)
    assert parsed, "expected at least one mirror bar"
    return parsed


def _find_sprites(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    sprite_grid = np.where(np.isin(grid, BAR_COLORS), bg, grid)
    sprites = find_components(
        sprite_grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert sprites, "expected at least one sprite"
    sprites = sorted(sprites, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return sprite_grid, sprites


def _bars_for_sprite(sprite: dict, bars: list[dict]) -> list[dict]:
    r0, c0, r1, c1 = map(int, sprite["bbox"])
    left_bars = []
    right_bars = []
    for bar in bars:
        br0, bc0, br1, _bc1 = map(int, bar["bbox"])
        if not (br0 <= r0 and r1 <= br1):
            continue
        if bc0 < c0:
            left_bars.append(bar)
        elif bc0 > c1:
            right_bars.append(bar)

    assert left_bars or right_bars, "every sprite must have a row-aligned mirror chain"
    assert not (left_bars and right_bars), "a sprite's mirror chain must stay on one side"
    if right_bars:
        return sorted(right_bars, key=lambda bar: int(bar["bbox"][1]))
    return sorted(left_bars, key=lambda bar: int(bar["bbox"][1]), reverse=True)


def _reflect_interval(left: int, right: int, axis_col: int) -> tuple[int, int]:
    return 2 * axis_col - right, 2 * axis_col - left


def _simulate_chain(patch: np.ndarray, start_left: int, bars: list[dict]) -> tuple[np.ndarray, int]:
    cur_patch = np.array(patch, copy=True)
    left = int(start_left)
    right = left + patch.shape[1] - 1
    for bar in bars:
        axis_col = int(bar["bbox"][1])
        left, right = _reflect_interval(left, right, axis_col)
        if int(bar["color"]) == MIRROR:
            cur_patch = np.fliplr(cur_patch)
    return cur_patch, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each multicolor sprite through its row-aligned vertical mirror chain and keep only the final reflected copy.

    Input structure:
    - The background is uniform.
    - Each non-bar foreground object is one multicolor sprite.
    - Every sprite has a same-row chain of one-column vertical bars on exactly one side; bars use only grey or teal.

    Rule:
    - For each sprite, order its same-row bars from nearest to farthest on that side.
    - Reflect the sprite's current bounding box across each bar column in sequence.
    - Grey bars also flip the sprite pattern left-right, while teal bars move the sprite without flipping it.
    - Remove the original sprites, preserve all bars, and draw only each sprite's final pattern at its final reflected position.

    Key insight:
    - The final location is the composition of all position reflections, while the sprite orientation depends only on the parity of grey-bar flips.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    sprite_grid, sprites = _find_sprites(g, bg)
    bars = _find_bar_components(g, bg)

    out = _clear_sprites(g, sprite_grid, bg)
    used_bars: set[tuple[int, int, int, int, int]] = set()

    for sprite in sprites:
        r0, c0, _r1, _c1 = map(int, sprite["bbox"])
        patch = crop_bbox(sprite_grid, sprite["bbox"])
        chain = _bars_for_sprite(sprite, bars)
        for bar in chain:
            used_bars.add(_bar_key(bar))
        final_patch, final_left = _simulate_chain(patch, c0, chain)
        paste(out, final_patch, r0, final_left, transparent=bg)

    assert used_bars == {_bar_key(bar) for bar in bars}, "every bar must belong to exactly one sprite chain"
    return out


def _sample_patch(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    for _ in range(100):
        patch = rng.choice(SPRITE_PALETTE, size=(h, w), replace=True).astype(int)
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    raise RuntimeError("failed to sample an asymmetric sprite patch")


def _sample_bar_colors(rng: np.random.Generator, count: int) -> list[int]:
    colors = rng.choice([MIRROR, WINDOW], size=count, replace=True).astype(int).tolist()
    if count >= 2 and len(set(colors)) == 1 and rng.random() < 0.75:
        idx = int(rng.integers(0, count))
        colors[idx] = WINDOW if colors[idx] == MIRROR else MIRROR
    return colors


def _sample_group(rng: np.random.Generator, width: int, forced_bar_count: int | None = None) -> dict:
    for _ in range(MAX_GROUP_ATTEMPTS):
        h = int(rng.integers(2, 6))
        w = int(rng.integers(2, 5))
        bar_count = forced_bar_count if forced_bar_count is not None else int(rng.integers(1, 5))
        gaps = [int(rng.integers(1, 4)) for _ in range(bar_count)]
        total_shift = sum(w + 2 * gap + 1 for gap in gaps)
        side = str(rng.choice(["left", "right"]))

        if side == "right":
            max_left = width - (w + total_shift)
            if max_left < 0:
                continue
            start_left = int(rng.integers(0, max_left + 1))
            cur_left = start_left
            bars = []
            for gap, color in zip(gaps, _sample_bar_colors(rng, bar_count)):
                col = cur_left + w + gap
                bars.append({"col": int(col), "color": int(color)})
                cur_left += w + 2 * gap + 1
        else:
            min_left = total_shift
            max_left = width - w
            if min_left > max_left:
                continue
            start_left = int(rng.integers(min_left, max_left + 1))
            cur_left = start_left
            bars = []
            for gap, color in zip(gaps, _sample_bar_colors(rng, bar_count)):
                col = cur_left - gap - 1
                bars.append({"col": int(col), "color": int(color)})
                cur_left -= w + 2 * gap + 1

        patch = _sample_patch(rng, h, w)
        final_patch, final_left = _simulate_chain(
            patch,
            start_left,
            [{"bbox": (0, bar["col"], h - 1, bar["col"]), "color": bar["color"]} for bar in bars],
        )
        if final_left < 0 or final_left + w > width:
            continue

        grey_count = sum(bar["color"] == MIRROR for bar in bars)
        expected_patch = np.fliplr(patch) if grey_count % 2 else patch
        if not np.array_equal(final_patch, expected_patch):
            continue

        return {
            "height": h,
            "width": w,
            "start_left": start_left,
            "bars": bars,
            "patch": patch,
            "side": side,
            "grey_count": grey_count,
            "final_left": final_left,
            "bar_count": bar_count,
        }
    raise RuntimeError("failed to sample a valid mirror-chain group")


def _sample_bar_count_plan(rng: np.random.Generator, group_count: int) -> list[int | None]:
    plan: list[int | None] = [None] * group_count
    order = list(range(group_count))
    rng.shuffle(order)
    plan[order[0]] = 1
    if group_count >= 2:
        plan[order[1]] = int(rng.integers(2, 5))
    return plan


def _compose_rows(rng: np.random.Generator, groups: list[dict]) -> tuple[int, list[int]] | None:
    top_margin = int(rng.integers(0, 2))
    bottom_margin = int(rng.integers(0, 2))
    gaps = [int(rng.integers(1, 4)) for _ in range(max(0, len(groups) - 1))]
    total_height = top_margin + bottom_margin + sum(group["height"] for group in groups) + sum(gaps)
    if total_height > 30:
        return None

    tops = []
    row = top_margin
    for idx, group in enumerate(groups):
        tops.append(row)
        row += group["height"]
        if idx < len(gaps):
            row += gaps[idx]
    return total_height, tops


def _render_case(width: int, height: int, groups: list[dict], tops: list[int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for group, top in zip(groups, tops):
        paste(inp, group["patch"], top, group["start_left"])
        for bar in group["bars"]:
            inp[top : top + group["height"], bar["col"]] = int(bar["color"])
            out[top : top + group["height"], bar["col"]] = int(bar["color"])
        final_patch = np.fliplr(group["patch"]) if group["grey_count"] % 2 else group["patch"]
        out[top : top + group["height"], group["final_left"] : group["final_left"] + group["width"]] = final_patch
    return inp, out


def _build_case(rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(MAX_CASE_ATTEMPTS):
        width = int(rng.integers(18, 31))
        group_count = int(rng.integers(2, 5))
        plan = _sample_bar_count_plan(rng, group_count)
        groups = []
        try:
            for forced_bar_count in plan:
                groups.append(_sample_group(rng, width, forced_bar_count=forced_bar_count))
        except RuntimeError:
            continue

        rows = _compose_rows(rng, groups)
        if rows is None:
            continue
        height, tops = rows
        inp, expected = _render_case(width, height, groups, tops)

        meta_groups = []
        for group, top in zip(groups, tops):
            meta_groups.append(
                {
                    "top": int(top),
                    "height": int(group["height"]),
                    "width": int(group["width"]),
                    "start_left": int(group["start_left"]),
                    "final_left": int(group["final_left"]),
                    "grey_count": int(group["grey_count"]),
                    "bar_count": int(group["bar_count"]),
                    "side": group["side"],
                    "bars": [dict(col=int(bar["col"]), color=int(bar["color"])) for bar in group["bars"]],
                }
            )

        return inp, expected, {"groups": meta_groups}
    raise RuntimeError("failed to generate a valid ARC input")


def generate_input() -> np.ndarray:
    return _build_case()[0]
