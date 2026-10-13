from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


BACKGROUND = 0
DIVIDER_COLOR = 2
SPRITE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_RETRIES = 300


def _mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


_TEMPLATES = [
    {"name": "sym_plus", "mask": _mask((".#.", "###", ".#.")), "symmetric": True},
    {"name": "sym_pillar", "mask": _mask(("###", ".#.", ".#.")), "symmetric": True},
    {"name": "sym_arch", "mask": _mask(("#.#", "###")), "symmetric": True},
    {"name": "sym_barbell", "mask": _mask((".##.", "####", ".##.")), "symmetric": True},
    {"name": "sym_line", "mask": _mask(("#", "#", "#", "#")), "symmetric": True},
    {"name": "sym_t", "mask": _mask(("###", ".#.")), "symmetric": True},
    {"name": "asym_stair", "mask": _mask(("#..", "##.", ".##")), "symmetric": False},
    {"name": "asym_hook", "mask": _mask(("###", "#..")), "symmetric": False},
    {"name": "asym_skew", "mask": _mask(("##.", ".#.", ".##")), "symmetric": False},
    {"name": "asym_l", "mask": _mask(("#.", "#.", "##")), "symmetric": False},
    {"name": "asym_zig4", "mask": _mask(("##..", ".##.", "..##")), "symmetric": False},
    {"name": "asym_tilt", "mask": _mask((".##", "##.", ".#.")), "symmetric": False},
]
for _template in _TEMPLATES:
    _template["h"], _template["w"] = _template["mask"].shape
    _template["area"] = int(_template["mask"].sum())

_TEMPLATE_BY_NAME = {template["name"]: template for template in _TEMPLATES}
_PAIR_CHOICES = [
    ("sym_plus", "asym_stair"),
    ("sym_pillar", "asym_skew"),
    ("sym_arch", "asym_tilt"),
]


def _divider_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        vals = set(map(int, g[r, :]))
        if len(vals) == 1:
            color = next(iter(vals))
            if int(color) != int(bg):
                rows.append((int(r), int(color)))
    assert len(rows) == 1, "expected exactly one solid non-background divider row"
    return rows[0]


def _component_patch(g: np.ndarray, comp: dict, bg: int) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(g, (r0, c0, r1, c1))
    mask = np.zeros_like(patch, dtype=bool)
    for rr, cc in comp["coords"]:
        mask[int(rr) - r0, int(cc) - c0] = True
    out = np.full_like(patch, int(bg))
    out[mask] = int(comp["color"])
    return out, mask


def _is_vertically_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _paint_first_wins(out: np.ndarray, claimed: np.ndarray, coords: np.ndarray, color: int) -> None:
    H, W = out.shape
    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        if 0 <= r < H and 0 <= c < W and not claimed[r, c] and int(out[r, c]) == BACKGROUND:
            out[r, c] = int(color)
            claimed[r, c] = True


def _build_expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    divider_r, divider_color = _divider_row(g, bg)
    assert np.all(g[divider_r + 1 :, :] == bg), "below the divider must start empty"

    sprite_colors = sorted(set(map(int, g.ravel())) - {bg, divider_color})
    assert sprite_colors, "expected foreground sprites above the divider"
    sprites = find_components(
        g,
        bg=bg,
        colors=sprite_colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert sprites, "expected at least one sprite"

    ordered = sorted(sprites, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0]), int(comp["color"])))
    out = g.copy()
    out[(g != bg) & (g != divider_color)] = bg
    claimed = np.zeros_like(g, dtype=bool)

    for comp in ordered:
        coords = np.asarray(comp["coords"], dtype=int)
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r1 < divider_r - 1, "sprites must not touch the divider"
        assert 3 <= int(coords.shape[0]) <= 8, "sprite area must be between 3 and 8"
        assert c1 - c0 + 1 <= 4, "sprite bbox width must be at most 4"
        _patch, mask = _component_patch(g, comp, bg)
        symmetric = _is_vertically_symmetric(mask)

        if symmetric:
            shifted = coords.copy()
            shifted[:, 0] += H - 1 - r1
            out[divider_r, np.unique(coords[:, 1]).astype(int)] = bg
            _paint_first_wins(out, claimed, shifted, color)
        else:
            shifted = coords.copy()
            shifted[:, 0] += divider_r - 1 - r1
            _paint_first_wins(out, claimed, shifted, color)

            mirror_mask = np.fliplr(mask)
            mirror_coords = np.argwhere(mirror_mask)
            mirror_coords[:, 0] += divider_r + 1
            mirror_coords[:, 1] += c0
            _paint_first_wins(out, claimed, mirror_coords, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each sprite through a symmetry gate so symmetric ones fall through and asymmetric ones stop and mirror below.

    Input structure:
    - The background is black and one non-background color forms a single full-width horizontal divider row.
    - Several 4-connected monochromatic sprites with distinct visible shapes sit above that divider, with empty space below it.
    - Each sprite fits a narrow bounding box, and its left-to-right placement determines processing order.

    Rule:
    - For each sprite, inspect its bbox-local mask and test left-right symmetry about that bbox's vertical midline.
    - A symmetric sprite is removed from its start position, translated straight down until its bottom row reaches the grid bottom, and every divider cell in its occupied columns is punched out to background.
    - An asymmetric sprite is removed from its start position, translated straight down until its bottom row is just above the divider, then its bbox-local mask is mirrored horizontally and drawn immediately below the divider in the same columns.
    - Process sprites from smaller bbox-left column to larger; if a target pixel is already occupied by an earlier processed sprite result, keep the earlier color and skip the later conflicting pixel.

    Key insight:
    - Symmetry is the gate condition rather than the thing being completed: it decides whether a sprite passes through the divider or gets stopped and leaves a mirrored reflection on the far side.
    """
    return _build_expected_output(grid)


def _flip_ud(mask: np.ndarray) -> np.ndarray:
    return np.flipud(mask).copy()


def _random_template(name: str) -> dict:
    template = _TEMPLATE_BY_NAME[name]
    mask = template["mask"].copy()
    if random.random() < 0.5:
        mask = _flip_ud(mask)
    return {
        "name": template["name"],
        "mask": mask,
        "symmetric": bool(template["symmetric"]),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "area": int(mask.sum()),
    }


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _place_starts(widths: list[int], grid_width: int) -> list[int]:
    base = sum(widths) + max(0, len(widths) - 1)
    slack = grid_width - base
    assert slack >= 0
    extras = _composition(slack, len(widths) + 1)
    starts: list[int] = []
    cur = extras[0]
    for idx, width in enumerate(widths):
        starts.append(cur)
        cur += width
        if idx < len(widths) - 1:
            cur += 1 + extras[idx + 1]
    return starts


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = np.full((h, w), BACKGROUND, dtype=int)
    patch[mask] = int(color)
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _template_selection(count: int) -> list[dict]:
    pair_sym, pair_asym = random.choice(_PAIR_CHOICES)
    chosen = [_random_template(pair_sym), _random_template(pair_asym)]

    sym_names = [t["name"] for t in _TEMPLATES if t["symmetric"]]
    asym_names = [t["name"] for t in _TEMPLATES if not t["symmetric"]]
    used = {pair_sym, pair_asym}

    while len(chosen) < count:
        want_sym = random.random() < 0.5
        pool = sym_names if want_sym else asym_names
        available = [name for name in pool if name not in used]
        if not available:
            available = pool
        name = random.choice(available)
        chosen.append(_random_template(name))
        used.add(name)

    random.shuffle(chosen)
    return chosen


def _analyze_sprite_columns(sprite: dict) -> tuple[int, int]:
    cols = np.argwhere(sprite["mask"])[:, 1]
    return int(cols.min()), int(cols.max())


def _expected_from_descriptors(H: int, W: int, divider_r: int, sprites: list[dict]) -> np.ndarray:
    out = np.full((H, W), BACKGROUND, dtype=int)
    out[divider_r, :] = DIVIDER_COLOR
    claimed = np.zeros((H, W), dtype=bool)

    for sprite in sorted(sprites, key=lambda item: (int(item["left"]), int(item["top"]))):
        mask = sprite["mask"]
        color = int(sprite["color"])
        top = int(sprite["top"])
        left = int(sprite["left"])
        h = int(mask.shape[0])
        occupied_local = np.argwhere(mask)

        if sprite["symmetric"]:
            out[divider_r, left + np.unique(occupied_local[:, 1]).astype(int)] = BACKGROUND
            shifted = occupied_local.copy()
            shifted[:, 0] += H - h
            shifted[:, 1] += left
            _paint_first_wins(out, claimed, shifted, color)
        else:
            shifted = occupied_local.copy()
            shifted[:, 0] += divider_r - h
            shifted[:, 1] += left
            _paint_first_wins(out, claimed, shifted, color)

            mirror_coords = np.argwhere(np.fliplr(mask))
            mirror_coords[:, 0] += divider_r + 1
            mirror_coords[:, 1] += left
            _paint_first_wins(out, claimed, mirror_coords, color)

    return out


def _validate_scene(inp: np.ndarray, sprites: list[dict], expected: np.ndarray) -> None:
    H, W = inp.shape
    assert H <= 30 and W <= 30

    divider_r, divider_color = _divider_row(inp, BACKGROUND)
    assert divider_color == DIVIDER_COLOR
    assert np.all(inp[divider_r + 1 :, :] == BACKGROUND), "input below divider must be empty"
    assert 3 <= len(sprites) <= 5

    comps = find_components(
        inp,
        bg=BACKGROUND,
        colors=SPRITE_COLORS,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(comps) == len(sprites)

    seen_ranges: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    branch_counts = {True: 0, False: 0}
    pair_ok = False

    for sprite in sprites:
        color = int(sprite["color"])
        mask = np.asarray(sprite["mask"], dtype=bool)
        h, w = mask.shape
        top = int(sprite["top"])
        left = int(sprite["left"])
        area = int(mask.sum())
        assert 3 <= area <= 8
        assert w <= 4
        assert top + h - 1 < divider_r - 1
        assert color not in seen_colors
        seen_colors.add(color)
        branch_counts[bool(sprite["symmetric"])] += 1

        c0, c1 = left, left + w - 1
        for prev0, prev1 in seen_ranges:
            assert c1 < prev0 - 1 or c0 > prev1 + 1, "sprite column ranges need a blank column gap"
        seen_ranges.append((c0, c1))

    assert branch_counts[True] >= 1 and branch_counts[False] >= 1
    assert any(sprite["symmetric"] and sprite["mask"].shape[1] > 1 for sprite in sprites)

    for i, a in enumerate(sprites):
        for b in sprites[i + 1 :]:
            if a["symmetric"] != b["symmetric"] and a["area"] == b["area"] and a["w"] == b["w"]:
                pair_ok = True
    assert pair_ok, "need at least one symmetric/asymmetric width-area matched pair"

    computed = main(inp)
    assert np.array_equal(computed, expected), "solver output must match explicit scene construction"

    hole_cols = set()
    for sprite in sprites:
        if sprite["symmetric"]:
            local_cols = np.unique(np.argwhere(sprite["mask"])[:, 1]).astype(int)
            hole_cols.update(int(sprite["left"]) + int(col) for col in local_cols)
    for c in range(W):
        want = BACKGROUND if c in hole_cols else DIVIDER_COLOR
        assert int(expected[divider_r, c]) == want

    for sprite in sprites:
        left = int(sprite["left"])
        color = int(sprite["color"])
        mask = sprite["mask"]
        h, w = mask.shape
        if sprite["symmetric"]:
            patch = expected[H - h : H, left : left + w]
            assert np.array_equal(patch == color, mask)
        else:
            top_patch = expected[divider_r - h : divider_r, left : left + w]
            bottom_patch = expected[divider_r + 1 : divider_r + 1 + h, left : left + w]
            assert np.array_equal(top_patch == color, mask)
            assert np.array_equal(bottom_patch == color, np.fliplr(mask))


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    templates = _template_selection(count)
    widths = [int(template["w"]) for template in templates]
    max_h = max(int(template["h"]) for template in templates)

    min_width = sum(widths) + max(0, count - 1)
    max_width = min(26, max(16, min_width + 6))
    if min_width > max_width:
        return None
    W = random.randint(min_width, max_width)

    min_height = max(13, 2 * max_h + 4)
    max_height = 19
    if min_height > max_height:
        return None
    H = random.randint(min_height, max_height)

    divider_min = max(max_h + 2, H // 2)
    divider_max = H - max_h - 2
    if divider_min > divider_max:
        return None
    divider_r = random.randint(divider_min, divider_max)

    starts = _place_starts(widths, W)
    colors = random.sample(SPRITE_COLORS, count)

    sprites: list[dict] = []
    for idx, template in enumerate(templates):
        h = int(template["h"])
        top_max = divider_r - h - 2
        if top_max < 0:
            return None
        sprites.append(
            {
                "name": template["name"],
                "mask": template["mask"],
                "symmetric": bool(template["symmetric"]),
                "h": h,
                "w": int(template["w"]),
                "area": int(template["area"]),
                "top": random.randint(0, top_max),
                "left": int(starts[idx]),
                "color": int(colors[idx]),
            }
        )

    inp = np.full((H, W), BACKGROUND, dtype=int)
    inp[divider_r, :] = DIVIDER_COLOR
    for sprite in sprites:
        _paint_mask(inp, sprite["mask"], int(sprite["top"]), int(sprite["left"]), int(sprite["color"]))

    expected = _expected_from_descriptors(H, W, divider_r, sprites)
    if np.array_equal(inp, expected):
        return None

    _validate_scene(inp, sprites, expected)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid symmetry-gate scene")
