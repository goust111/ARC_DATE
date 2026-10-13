from __future__ import annotations

from collections import Counter
from itertools import permutations
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, paste, random_free_location


BG = Color.BLACK
SEP = Color.GREY

_CODEBOOK: dict[tuple[tuple[int, ...], ...], int] = {
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)): Color.RED,
    ((1, 0, 0), (1, 1, 0), (1, 1, 1)): Color.YELLOW,
    ((1, 0, 1), (0, 1, 0), (1, 0, 1)): Color.TEAL,
}
_SYMBOL_BY_COLOR = {
    color: np.array(mask, dtype=int) for mask, color in _CODEBOOK.items()
}
_ANCHOR_COLORS = tuple(_SYMBOL_BY_COLOR)

_SPRITE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("AB.", "BBB", ".BB"),
    (".AB.", "BBBB", ".BB.", ".BB."),
    ("..A..", ".BBB.", "BBBBB", ".BCB.", ".CCC."),
    ("ABB..", "BBBB.", ".BCBB", "..CCB"),
    (".A..", "BBB.", ".BCB", ".CCB", "..BB"),
    ("..A.", ".BBB", "BBCB", ".CCC"),
)

_AUX_COLORS = tuple(c for c in Color.NOT_BLACK if c not in {SEP, *_ANCHOR_COLORS})
_GEN_RETRY_LIMIT = 600
_GEN_CALL_COUNT = 0
_GEN_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _groups_of_consecutive(rows: list[int]) -> list[list[int]]:
    if not rows:
        return []
    groups = [[int(rows[0])]]
    for row in rows[1:]:
        row = int(row)
        if row == groups[-1][-1] + 1:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    return int(cols[0])


def _anchor_color_and_pos(grid: np.ndarray, coords: np.ndarray) -> tuple[int, tuple[int, int]]:
    colors = [int(grid[r, c]) for r, c in coords]
    counts = Counter(colors)
    min_count = min(counts.values())
    rare = [color for color, count in counts.items() if count == min_count]
    assert min_count == 1 and len(rare) == 1, "expected a unique singleton anchor color"
    anchor_color = int(rare[0])
    anchor_pos = tuple(map(int, coords[np.where([color == anchor_color for color in colors])[0][0]]))
    return anchor_color, anchor_pos


def _extract_sprites(left: np.ndarray) -> dict[int, tuple[np.ndarray, tuple[int, int]]]:
    sprites: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for comp in find_components(left, bg=BG, connectivity=4, monochromatic=False, return_format="dict"):
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = left[r0 : r1 + 1, c0 : c1 + 1].copy()
        anchor_color, (ar, ac) = _anchor_color_and_pos(left, coords)
        assert anchor_color not in sprites, "anchor colors must be unique across sprites"
        sprites[anchor_color] = (patch, (int(ar - r0), int(ac - c0)))
    assert 2 <= len(sprites) <= 3, "expected 2-3 source sprites"
    return sprites


def _panel_rows(right: np.ndarray) -> list[int]:
    active = [r for r in range(right.shape[0]) if np.any(right[r] != BG)]
    groups = _groups_of_consecutive(active)
    starts: list[int] = []
    for group in groups:
        assert len(group) == 3, "each target panel must be exactly 3 rows tall"
        starts.append(int(group[0]))
    assert 2 <= len(starts) <= 3, "expected 2-3 target panels"
    return starts


def _decode_panels(right: np.ndarray) -> list[tuple[int, int]]:
    starts = _panel_rows(right)
    center_c = right.shape[1] // 2
    assert 1 <= center_c - 1 and center_c + 1 < right.shape[1], "right region must center a 3x3 symbol"
    decoded: list[tuple[int, int]] = []
    for r0 in starts:
        panel = right[r0 : r0 + 3, :]
        assert set(map(int, np.unique(panel))) <= {BG, SEP}, "panels must contain only black and grey"
        symbol = (panel[:, center_c - 1 : center_c + 2] == SEP).astype(int)
        outside = panel.copy()
        outside[:, center_c - 1 : center_c + 2] = BG
        assert not np.any(outside == SEP), "grey symbol must stay inside the centered 3x3 window"
        key = _mask_key(symbol)
        assert key in _CODEBOOK, "encountered an unknown symbol"
        decoded.append((int(r0), int(_CODEBOOK[key])))
    return decoded


def _stamp_sequence(
    right_shape: tuple[int, int],
    decoded_panels: list[tuple[int, int]],
    sprites: dict[int, tuple[np.ndarray, tuple[int, int]]],
) -> np.ndarray:
    out = np.full(right_shape, BG, dtype=int)
    center_c = right_shape[1] // 2
    for r0, anchor_color in decoded_panels:
        patch, (ar, ac) = sprites[anchor_color]
        top = int(r0 + 1 - ar)
        left = int(center_c - ac)
        paste(out, patch, top, left, transparent=BG, clip=True)
        out[r0 + 1, center_c] = anchor_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each centered 3x3 grey symbol into an anchor color and stamp the matching source sprite into the target strip.

    Input structure:
    - A full-height grey separator column splits the black background grid into a left source region and a right target region.
    - The left region contains 2-3 small multicolor 4-connected sprites; each sprite has exactly one singleton color, which is its anchor color.
    - The right region contains 2-3 horizontal panels of height 3, separated by black rows, and each panel holds one centered 3x3 black/grey symbol mask.

    Rule:
    - Find the separator column, extract the left sprites, and identify each sprite's anchor as its unique least-frequent color.
    - Decode every centered 3x3 grey symbol with the task's fixed symbol-to-color codebook.
    - For each panel from top to bottom, copy the sprite whose anchor color matches the decoded color so that the sprite's anchor pixel lands on the panel center; clip anything that falls outside the right region.
    - Return only the right region after stamping, with later panels overwriting earlier overlaps.

    Key insight:
    - The grey symbols are not the final output objects; they are lookup keys that select which anchored source sprite gets stamped into each target panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_c = _separator_col(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    assert right.shape[1] >= 3 and right.shape[1] % 2 == 1, "target strip must have an odd width of at least 3"

    sprites = _extract_sprites(left)
    decoded_panels = _decode_panels(right)
    assert {color for _r0, color in decoded_panels} == set(sprites), "decoded colors must match source anchors one-to-one"
    return _stamp_sequence(right.shape, decoded_panels, sprites)


def _transform_chars(template: tuple[str, ...], transform_id: int) -> np.ndarray:
    arr = np.array([list(row) for row in template], dtype="<U1")
    transform_id = int(transform_id)
    if transform_id >= 4:
        arr = np.fliplr(arr)
        transform_id -= 4
    return np.rot90(arr, k=transform_id)


def _render_sprite(
    template: tuple[str, ...],
    anchor_color: int,
    body_color: int,
    accent_color: int | None,
    transform_id: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    chars = _transform_chars(template, transform_id)
    patch = np.full(chars.shape, BG, dtype=int)
    patch[chars == "A"] = int(anchor_color)
    patch[chars == "B"] = int(body_color)
    if accent_color is not None:
        patch[chars == "C"] = int(accent_color)
    anchor_cells = np.argwhere(chars == "A")
    assert anchor_cells.shape == (1, 2), "each template must contain exactly one anchor cell"
    return patch, tuple(map(int, anchor_cells[0]))


def _panel_starts(n_panels: int, min_height: int) -> tuple[int, list[int]]:
    gap_choices = [1] * max(1, n_panels - 1)
    if n_panels > 1:
        for i in range(n_panels - 1):
            if random.random() < 0.35:
                gap_choices[i] = 2
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    height = top_margin + 3 * n_panels + sum(gap_choices) + bottom_margin
    if height < min_height:
        bottom_margin += min_height - height
        height = min_height
    starts = []
    row = top_margin
    for i in range(n_panels):
        starts.append(int(row))
        row += 3
        if i < n_panels - 1:
            row += gap_choices[i]
    return int(height), starts


def _choose_panel_order(anchor_colors: list[int], source_order: list[int]) -> list[int]:
    orders = [list(order) for order in permutations(anchor_colors) if list(order) != source_order]
    assert orders, "need a non-identity panel order"
    random.shuffle(orders)
    return orders[0]


def _place_sprites(
    height: int,
    left_width: int,
    rendered: list[dict[str, object]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    left = np.full((height, left_width), BG, dtype=int)
    placed: list[dict[str, object]] = []
    for sprite in sorted(rendered, key=lambda rec: int(np.count_nonzero(rec["patch"])), reverse=True):
        patch = np.array(sprite["patch"], dtype=int)
        try:
            top, left_c = random_free_location(
                left,
                patch,
                transparent=BG,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(left, patch, top, left_c, transparent=BG)
        ar, ac = sprite["anchor"]
        placed.append(
            {
                **sprite,
                "top": int(top),
                "left": int(left_c),
                "anchor_global": (int(top + ar), int(left_c + ac)),
            }
        )
    return left, placed


def _render_input_grid(
    left: np.ndarray,
    right_width: int,
    panel_starts: list[int],
    panel_colors: list[int],
) -> np.ndarray:
    height = left.shape[0]
    grid = np.full((height, left.shape[1] + 1 + right_width), BG, dtype=int)
    grid[:, : left.shape[1]] = left
    grid[:, left.shape[1]] = SEP
    center_c = right_width // 2
    right_offset = left.shape[1] + 1
    for r0, color in zip(panel_starts, panel_colors):
        symbol = _SYMBOL_BY_COLOR[int(color)]
        for rr in range(3):
            for cc in range(3):
                if int(symbol[rr, cc]):
                    grid[r0 + rr, right_offset + center_c - 1 + cc] = SEP
    return grid


def _naive_topdown_output(
    right_shape: tuple[int, int],
    panel_starts: list[int],
    source_order: list[int],
    sprites: dict[int, tuple[np.ndarray, tuple[int, int]]],
) -> np.ndarray:
    decoded = [(int(r0), int(color)) for r0, color in zip(panel_starts, source_order)]
    return _stamp_sequence(right_shape, decoded, sprites)


def _validate_instance(grid: np.ndarray, meta: dict[str, object]) -> bool:
    sep_c = _separator_col(grid)
    left = grid[:, :sep_c]
    right = grid[:, sep_c + 1 :]
    sprites = _extract_sprites(left)
    decoded = _decode_panels(right)
    panel_starts = [r0 for r0, _color in decoded]
    panel_colors = [color for _r0, color in decoded]

    expected = _stamp_sequence(right.shape, decoded, sprites)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if panel_colors != list(meta["panel_colors"]):
        return False
    if panel_starts != list(meta["panel_starts"]):
        return False

    source_order = list(meta["source_order"])
    if panel_colors == source_order:
        return False

    naive = _naive_topdown_output(right.shape, panel_starts, source_order, sprites)
    if np.array_equal(naive, expected):
        return False

    center_c = right.shape[1] // 2
    for r0, color in decoded:
        if int(expected[r0 + 1, center_c]) != int(color):
            return False

    if int(np.count_nonzero(expected)) <= len(decoded):
        return False

    if len(set(panel_colors)) != len(panel_colors):
        return False
    if set(panel_colors) != set(sprites):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    n_objects = 3 if (_GEN_CALL_COUNT % 3 != 0) else 2
    anchor_colors = random.sample(list(_ANCHOR_COLORS), n_objects)
    template_indices = random.sample(range(len(_SPRITE_TEMPLATES)), n_objects)

    body_colors = random.sample(list(_AUX_COLORS), n_objects)
    rendered: list[dict[str, object]] = []
    for anchor_color, template_idx, body_color in zip(anchor_colors, template_indices, body_colors):
        template = _SPRITE_TEMPLATES[int(template_idx)]
        template_arr = np.array([list(row) for row in template], dtype="<U1")
        needs_accent = bool(np.any(template_arr == "C"))
        accent_choices = [c for c in _AUX_COLORS if c != body_color]
        accent_color = random.choice(accent_choices) if needs_accent else None
        patch, anchor = _render_sprite(
            template,
            anchor_color=anchor_color,
            body_color=body_color,
            accent_color=accent_color,
            transform_id=random.randrange(8),
        )
        rendered.append(
            {
                "anchor_color": int(anchor_color),
                "patch": patch,
                "anchor": anchor,
            }
        )

    max_sprite_h = max(int(rec["patch"].shape[0]) for rec in rendered)
    max_sprite_w = max(int(rec["patch"].shape[1]) for rec in rendered)
    min_height = max(10, max_sprite_h + 4)
    height, panel_starts = _panel_starts(n_objects, min_height=min_height)

    left_width = random.randint(max(8, max_sprite_w + 3), 12)
    right_width = random.choice((5, 7))
    if max(left_width + 1 + right_width, height) > 30:
        return None

    placed_result = _place_sprites(height, left_width, rendered)
    if placed_result is None:
        return None
    left, placed = placed_result

    source_order = [
        int(rec["anchor_color"])
        for rec in sorted(
            placed,
            key=lambda rec: (int(rec["anchor_global"][0]), int(rec["anchor_global"][1])),
        )
    ]
    panel_colors = _choose_panel_order(anchor_colors, source_order)

    grid = _render_input_grid(left, right_width, panel_starts, panel_colors)
    meta = {
        "panel_starts": panel_starts,
        "panel_colors": panel_colors,
        "source_order": source_order,
    }
    return grid if _validate_instance(grid, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            out = main(grid)
            sig = (tuple(map(int, out.shape)), out.tobytes())
            if sig in _GEN_SEEN_OUTPUTS:
                continue
            _GEN_SEEN_OUTPUTS.add(sig)
            return grid
    raise RuntimeError("failed to generate a valid symbol-decoding stamp input")
