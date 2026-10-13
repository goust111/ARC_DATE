from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = spec_from_file_location("_shared_arc_utils_024_9", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
neighbors4 = _ARC.neighbors4


BG = Color.BLACK
LEFT_COLOR = Color.BLUE
RIGHT_COLOR = Color.RED
MARKER_COLOR = Color.GREEN
SEP_COLOR = Color.GREY
SPRITE_BODY_COLORS = (Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

SPRITE_TEMPLATES = (
    (((0, 0), (0, 1), (1, 0)), (0, 1)),
    (((0, 0), (1, 0), (1, 1)), (1, 0)),
    (((0, 0), (1, 0), (2, 0)), (1, 0)),
    (((0, 0), (0, 1), (0, 2), (1, 1)), (1, 1)),
    (((0, 1), (1, 0), (1, 1), (2, 1)), (1, 1)),
    (((0, 0), (1, 0), (1, 1), (1, 2)), (1, 1)),
    (((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)), (1, 1)),
)

PANEL_BLOCKS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
)

SIDE_RANK = {"top": 0, "bottom": 1, "left": 2, "right": 3}
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Anchor:
    side: str
    axis: str
    r: int
    c: int
    sort_key: tuple[int, int]


@dataclass(frozen=True)
class Sprite:
    coords_in: tuple[tuple[int, int], ...]
    colors: tuple[int, ...]
    marker_in: tuple[int, int]
    coords_out: tuple[tuple[int, int], ...]
    marker_out: tuple[int, int]
    dr: int
    dc: int
    final_coords: tuple[tuple[int, int], ...]
    sort_key: tuple[int, int]


def _normalize_shape(cells: tuple[tuple[int, int], ...], marker: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm_cells = tuple(sorted((r - r0, c - c0) for r, c in cells))
    norm_marker = (marker[0] - r0, marker[1] - c0)
    return norm_cells, norm_marker


def _rotate_shape(cells: tuple[tuple[int, int], ...], marker: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    max_c = max(c for _, c in cells)
    rot_cells = tuple((c, max_c - r) for r, c in cells)
    rot_marker = (marker[1], max_c - marker[0])
    return _normalize_shape(rot_cells, rot_marker)


def _flip_shape(cells: tuple[tuple[int, int], ...], marker: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    max_c = max(c for _, c in cells)
    flip_cells = tuple((r, max_c - c) for r, c in cells)
    flip_marker = (marker[0], max_c - marker[1])
    return _normalize_shape(flip_cells, flip_marker)


def _random_sprite_shape() -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    cells, marker = random.choice(SPRITE_TEMPLATES)
    cells = tuple(cells)
    marker = tuple(marker)
    for _ in range(random.randint(0, 3)):
        cells, marker = _rotate_shape(cells, marker)
    if random.random() < 0.5:
        cells, marker = _flip_shape(cells, marker)
    return _normalize_shape(cells, marker)


def _random_panel_block() -> tuple[tuple[int, int], ...]:
    block = tuple(PANEL_BLOCKS[random.randrange(len(PANEL_BLOCKS))])
    if random.random() < 0.5:
        max_c = max(c for _, c in block)
        block = tuple((r, max_c - c) for r, c in block)
    if random.random() < 0.5:
        max_r = max(r for r, _ in block)
        block = tuple((max_r - r, c) for r, c in block)
    return tuple(sorted(block))


def _collapsed_col(c: int, sep_col: int) -> int:
    if c < sep_col:
        return c
    if c > sep_col:
        return c - 1
    raise ValueError("separator column cannot appear in a collapsed coordinate")


def _collapse_coords(coords: tuple[tuple[int, int], ...], sep_col: int) -> tuple[tuple[int, int], ...]:
    return tuple((r, _collapsed_col(c, sep_col)) for r, c in coords)


def _anchor_from_cell(r: int, c: int, height: int, width: int) -> Anchor:
    if c == 0:
        side = "left"
        axis = "v"
        sort_key = (r, c)
    elif c == width - 1:
        side = "right"
        axis = "v"
        sort_key = (r, c)
    elif r == 0:
        side = "top"
        axis = "h"
        sort_key = (c, r)
    else:
        side = "bottom"
        axis = "h"
        sort_key = (c, r)
    return Anchor(side=side, axis=axis, r=r, c=c, sort_key=sort_key)


def _border_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = [(0, c) for c in range(width)]
    cells.extend((height - 1, c) for c in range(width))
    cells.extend((r, 0) for r in range(1, height - 1))
    cells.extend((r, width - 1) for r in range(1, height - 1))
    return cells


def _is_isolated_border_anchor(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != MARKER_COLOR:
        return False
    height, width = grid.shape
    if not (r in (0, height - 1) or c in (0, width - 1)):
        return False
    for rr, cc in neighbors4(r, c):
        if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) != BG:
            return False
    return True


def _panel_rows(grid: np.ndarray, sep_col: int) -> tuple[int, int]:
    left_hits = np.any(grid[:, :sep_col] == LEFT_COLOR, axis=1)
    right_hits = np.any(grid[:, sep_col + 1 :] == RIGHT_COLOR, axis=1)
    rows = np.flatnonzero(left_hits | right_hits)
    assert rows.size >= 5, "panel region must be at least five rows high"
    r0 = int(rows.min())
    r1 = int(rows.max())
    for r in range(r0, r1 + 1):
        assert left_hits[r] or right_hits[r], "every panel row must contain visible panel foreground"
    return r0, r1


def _detect_anchors(grid: np.ndarray) -> list[Anchor]:
    height, width = grid.shape
    anchors = [
        _anchor_from_cell(r, c, height, width)
        for r, c in _border_cells(height, width)
        if _is_isolated_border_anchor(grid, r, c)
    ]
    assert anchors, "expected at least one border anchor"
    return anchors


def _sprite_grid(grid: np.ndarray, sep_col: int, anchors: list[Anchor]) -> np.ndarray:
    aux = np.array(grid, dtype=int, copy=True)
    r0, r1 = _panel_rows(aux, sep_col)
    aux[:, sep_col] = BG
    aux[r0 : r1 + 1, :sep_col][aux[r0 : r1 + 1, :sep_col] == LEFT_COLOR] = BG
    aux[r0 : r1 + 1, sep_col + 1 :][aux[r0 : r1 + 1, sep_col + 1 :] == RIGHT_COLOR] = BG
    for anchor in anchors:
        aux[anchor.r, anchor.c] = BG
    return aux


def _extract_sprites(grid: np.ndarray, sep_col: int, anchors: list[Anchor]) -> list[Sprite]:
    aux = _sprite_grid(grid, sep_col, anchors)
    height, width = aux.shape
    out: list[Sprite] = []
    for comp in find_components(aux, bg=BG, connectivity=4, monochromatic=False, return_format="dict"):
        coords = tuple(tuple(map(int, p)) for p in comp["coords"])
        if not (3 <= len(coords) <= 5):
            continue
        marker_cells = [coord for coord in coords if int(aux[coord]) == MARKER_COLOR]
        assert len(marker_cells) == 1, "each sprite must contain exactly one green marker pixel"
        colors = tuple(int(aux[r, c]) for r, c in coords)
        coords_out = _collapse_coords(coords, sep_col)
        marker_in = marker_cells[0]
        marker_out = (marker_in[0], _collapsed_col(marker_in[1], sep_col))
        assert 0 <= marker_out[0] < height and 0 <= marker_out[1] < width - 1
        out.append(
            Sprite(
                coords_in=coords,
                colors=colors,
                marker_in=marker_in,
                coords_out=coords_out,
                marker_out=marker_out,
                dr=0,
                dc=0,
                final_coords=coords_out,
                sort_key=(0, 0),
            )
        )
    assert out, "expected at least one sprite"
    return out


def _sort_sprites_for_side(sprites: list[Sprite], side: str) -> list[Sprite]:
    if side in {"top", "bottom"}:
        return sorted(sprites, key=lambda sprite: (sprite.marker_out[1], sprite.marker_out[0]))
    return sorted(sprites, key=lambda sprite: (sprite.marker_out[0], sprite.marker_out[1]))


def _sort_anchors(anchors: list[Anchor]) -> list[Anchor]:
    return sorted(anchors, key=lambda anchor: (SIDE_RANK[anchor.side], anchor.sort_key))


def _nearest_axis_value(values: list[int], source: int) -> int | None:
    ranked = sorted((abs(v - source), v) for v in values)
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0] and ranked[0][1] != ranked[1][1]:
        return None
    return int(ranked[0][1])


def _target_mask(grid: np.ndarray, sep_col: int) -> tuple[np.ndarray, tuple[int, int]]:
    r0, r1 = _panel_rows(grid, sep_col)
    left = grid[r0 : r1 + 1, :sep_col]
    right = grid[r0 : r1 + 1, sep_col + 1 :]
    mask = (left == BG) & (right == BG)
    assert int(mask.sum()) >= 3, "target intersection must contain at least three cells"
    coords = np.argwhere(mask)
    assert len(set(map(int, coords[:, 0]))) > 1, "target cells cannot all share one row"
    assert len(set(map(int, coords[:, 1]))) > 1, "target cells cannot all share one column"
    return mask, (r0, r1)


def _output_axis_targets(mask: np.ndarray, panel_rows: tuple[int, int], panel_width: int, axis: str) -> list[int]:
    coords = np.argwhere(mask)
    if axis == "h":
        logical_cols = sorted(set(int(c) for _, c in coords))
        values = logical_cols + [c + panel_width for c in logical_cols]
        return sorted(set(values))
    r0, _r1 = panel_rows
    return sorted(set(r0 + int(r) for r, _ in coords))


def _move_sprites(
    base_out: np.ndarray,
    sprites: list[Sprite],
    anchors: list[Anchor],
    target_axis_values: list[int],
) -> np.ndarray:
    out = np.array(base_out, copy=True)
    occupied = np.zeros_like(out, dtype=bool)
    if len(anchors) == 1:
        side = anchors[0].side
    else:
        unique_sides = {anchor.side for anchor in anchors}
        assert len(unique_sides) == 1, "multi-sprite cases require anchors on the same side for unique pairing"
        side = next(iter(unique_sides))

    axis = anchors[0].axis
    assert all(anchor.axis == axis for anchor in anchors), "all anchors must imply the same slide axis"
    ordered_sprites = _sort_sprites_for_side(sprites, side)
    ordered_anchors = _sort_anchors(anchors)
    assert len(ordered_sprites) == len(ordered_anchors), "anchor count must equal sprite count"

    for sprite, _anchor in zip(ordered_sprites, ordered_anchors):
        if axis == "h":
            target = _nearest_axis_value(target_axis_values, sprite.marker_out[1])
            assert target is not None, "sprite must have a unique nearest target column"
            dr, dc = 0, int(target - sprite.marker_out[1])
        else:
            target = _nearest_axis_value(target_axis_values, sprite.marker_out[0])
            assert target is not None, "sprite must have a unique nearest target row"
            dr, dc = int(target - sprite.marker_out[0]), 0

        for (r, c), color in zip(sprite.coords_out, sprite.colors):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and not occupied[rr, cc]:
                out[rr, cc] = int(color)
                occupied[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compute the two-panel blank intersection, then slide each marked sprite along the anchor-chosen axis to the nearest target coordinate.

    Input structure:
    - The grid has a solid grey separator column splitting equal-width blue and red side panels on a black background.
    - Blue foreground appears only in the left panel and red foreground only in the right panel; sprites are separate 4-connected multicolor objects with exactly one green marker pixel.
    - One or two isolated green border anchors lie on the grid boundary and determine the slide axis for the sprites.

    Rule:
    - Remove the separator column from the output coordinate frame.
    - Compare the two panel interiors cell by cell; wherever both panels are black, paint that position green in both panel copies of the separator-removed output, while leaving existing blue/red panel foreground unchanged.
    - For each sprite, use its matched green border anchor to choose the motion axis: top/bottom anchors mean horizontal sliding, left/right anchors (including corners) mean vertical sliding.
    - Slide each sprite only along that axis until its green marker reaches the unique nearest target row or column value; when multiple sprites share one side, process them in that side order and keep earlier sprite pixels on collisions.

    Key insight:
    - The panel comparison does not directly produce the final picture: it creates a target coordinate set, while the border anchors independently choose the one-dimensional motion constraint that tells each sprite how to use those coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert width % 2 == 1 and width >= 11, "expected equal-width side panels separated by one column"

    sep_col = width // 2
    assert np.all(g[:, sep_col] == SEP_COLOR), "expected a solid grey separator column"
    panel_width = sep_col

    anchors = _detect_anchors(g)
    mask, panel_rows = _target_mask(g, sep_col)
    sprites = _extract_sprites(g, sep_col, anchors)
    assert len(sprites) == len(anchors), "anchor count must equal sprite count"

    out = np.full((height, width - 1), BG, dtype=int)
    r0, r1 = panel_rows
    left = g[r0 : r1 + 1, :sep_col]
    right = g[r0 : r1 + 1, sep_col + 1 :]

    out[r0 : r1 + 1, :panel_width][mask] = MARKER_COLOR
    out[r0 : r1 + 1, panel_width:][mask] = MARKER_COLOR
    left_fg = left != BG
    right_fg = right != BG
    out[r0 : r1 + 1, :panel_width][left_fg] = left[left_fg]
    out[r0 : r1 + 1, panel_width:][right_fg] = right[right_fg]

    axis = anchors[0].axis
    target_axis_values = _output_axis_targets(mask, panel_rows, panel_width, axis)
    return _move_sprites(out, sprites, anchors, target_axis_values)


def _paint_panel(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    view = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    view[mask] = int(color)


def _panel_component_count(mask: np.ndarray, color: int) -> int:
    panel = np.zeros(mask.shape, dtype=int)
    panel[mask] = color
    return len(find_components(panel, bg=BG, connectivity=4, monochromatic=True))


def _random_panel_masks(panel_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    left = np.zeros((panel_h, panel_w), dtype=bool)
    right = np.zeros((panel_h, panel_w), dtype=bool)
    for mask in (left, right):
        block_count = random.randint(3, 6)
        for _ in range(block_count):
            block = _random_panel_block()
            h = max(r for r, _ in block) + 1
            w = max(c for _, c in block) + 1
            rr = random.randint(0, panel_h - h)
            cc = random.randint(0, panel_w - w)
            for dr, dc in block:
                mask[rr + dr, cc + dc] = True

    for r in range(panel_h):
        if not (left[r].any() or right[r].any()):
            col = random.randrange(panel_w)
            if random.random() < 0.5:
                left[r, col] = True
            else:
                right[r, col] = True

    target = (~left) & (~right)
    coords = np.argwhere(target)
    if coords.shape[0] < 3:
        return None
    if len(set(map(int, coords[:, 0]))) == 1 or len(set(map(int, coords[:, 1]))) == 1:
        return None
    if _panel_component_count(left, LEFT_COLOR) < 2 or _panel_component_count(right, RIGHT_COLOR) < 2:
        return None
    return left, right


def _sprite_candidate_records(
    shape_cells: tuple[tuple[int, int], ...],
    marker_rel: tuple[int, int],
    height: int,
    width: int,
    sep_col: int,
    extra_rows: range,
    axis: str,
    target_axis_values: list[int],
) -> list[dict]:
    candidates: list[dict] = []
    shape_h = max(r for r, _ in shape_cells) + 1
    shape_w = max(c for _, c in shape_cells) + 1
    row_values = [
        r
        for r in extra_rows
        if 1 <= r and r + shape_h - 1 <= height - 2
    ]
    for r0 in row_values:
        for c0 in range(1, width - shape_w):
            coords_in = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape_cells))
            cols = [c for _, c in coords_in]
            if any(c == sep_col for c in cols):
                continue
            if min(cols) < 1 or max(cols) > width - 2:
                continue
            left_side = max(cols) < sep_col
            right_side = min(cols) > sep_col
            if not (left_side or right_side):
                continue
            marker_in = (r0 + marker_rel[0], c0 + marker_rel[1])
            coords_out = _collapse_coords(coords_in, sep_col)
            marker_out = (marker_in[0], _collapsed_col(marker_in[1], sep_col))
            if axis == "h":
                target = _nearest_axis_value(target_axis_values, marker_out[1])
                if target is None:
                    continue
                dr, dc = 0, int(target - marker_out[1])
                sort_key = (marker_out[1], marker_out[0])
            else:
                target = _nearest_axis_value(target_axis_values, marker_out[0])
                if target is None:
                    continue
                dr, dc = int(target - marker_out[0]), 0
                sort_key = (marker_out[0], marker_out[1])
            final_coords = tuple(sorted((r + dr, c + dc) for r, c in coords_out))
            if min(r for r, _ in final_coords) < 0 or max(r for r, _ in final_coords) >= height:
                continue
            if min(c for _, c in final_coords) < 0 or max(c for _, c in final_coords) >= width - 1:
                continue
            if dr == 0 and dc == 0:
                continue
            candidates.append(
                {
                    "coords_in": coords_in,
                    "marker_in": marker_in,
                    "coords_out": coords_out,
                    "marker_out": marker_out,
                    "dr": dr,
                    "dc": dc,
                    "final_coords": final_coords,
                    "sort_key": sort_key,
                }
            )
    return candidates


def _disjoint(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    return set(a).isdisjoint(set(b))


def _make_sprite(record: dict, body_color: int, marker_color: int = MARKER_COLOR) -> Sprite:
    colors = tuple(marker_color if coord == record["marker_in"] else body_color for coord in record["coords_in"])
    return Sprite(
        coords_in=record["coords_in"],
        colors=colors,
        marker_in=record["marker_in"],
        coords_out=record["coords_out"],
        marker_out=record["marker_out"],
        dr=int(record["dr"]),
        dc=int(record["dc"]),
        final_coords=record["final_coords"],
        sort_key=record["sort_key"],
    )


def _anchor_slots(grid: np.ndarray, side: str, count: int) -> list[tuple[int, int]] | None:
    height, width = grid.shape
    candidates: list[tuple[int, int]] = []
    if side == "top":
        iterable = [(0, c) for c in range(1, width - 1)]
    elif side == "bottom":
        iterable = [(height - 1, c) for c in range(1, width - 1)]
    elif side == "left":
        iterable = [(r, 0) for r in range(1, height - 1)]
    else:
        iterable = [(r, width - 1) for r in range(1, height - 1)]

    for r, c in iterable:
        if int(grid[r, c]) != BG:
            continue
        ok = True
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) != BG:
                ok = False
                break
        if ok:
            candidates.append((r, c))

    if count == 1:
        return [random.choice(candidates)] if candidates else None
    pairs = [
        [a, b]
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2
    ]
    return random.choice(pairs) if pairs else None


def _build_expected_output(
    height: int,
    panel_w: int,
    panel_top: int,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    sprites: list[Sprite],
    anchors: list[Anchor],
) -> np.ndarray:
    out = np.full((height, panel_w * 2), BG, dtype=int)
    target = (~left_mask) & (~right_mask)
    out[panel_top : panel_top + left_mask.shape[0], :panel_w][target] = MARKER_COLOR
    out[panel_top : panel_top + left_mask.shape[0], panel_w:][target] = MARKER_COLOR
    out[panel_top : panel_top + left_mask.shape[0], :panel_w][left_mask] = LEFT_COLOR
    out[panel_top : panel_top + left_mask.shape[0], panel_w:][right_mask] = RIGHT_COLOR

    axis = anchors[0].axis
    target_axis_values = _output_axis_targets(target, (panel_top, panel_top + left_mask.shape[0] - 1), panel_w, axis)
    return _move_sprites(out, sprites, anchors, target_axis_values)


def _gen_attempt(anchor_side_hint: str | None = None) -> np.ndarray | None:
    panel_w = random.randint(5, 8)
    panel_h = random.randint(5, 8)
    extra_h = random.randint(4, 6)
    height = panel_h + extra_h
    width = panel_w * 2 + 1
    sep_col = panel_w

    anchor_side = anchor_side_hint or random.choice(("top", "bottom", "left", "right"))
    extra_side = anchor_side if anchor_side in {"top", "bottom"} else random.choice(("top", "bottom"))
    panel_top = extra_h if extra_side == "top" else 0
    panel_bottom = panel_top + panel_h - 1
    extra_rows = range(0, extra_h) if extra_side == "top" else range(panel_bottom + 1, height)

    masks = _random_panel_masks(panel_h, panel_w)
    if masks is None:
        return None
    left_mask, right_mask = masks

    axis = "h" if anchor_side in {"top", "bottom"} else "v"
    sprite_count = random.choice((1, 2))
    want_overlap = sprite_count == 2 and random.random() < 0.35

    target_axis_values = _output_axis_targets((~left_mask) & (~right_mask), (panel_top, panel_bottom), panel_w, axis)
    logical_axis_values = sorted(set(int(v) for v in target_axis_values))
    if len(logical_axis_values) < 2:
        return None

    records_by_shape = []
    for _ in range(sprite_count):
        shape_cells, marker_rel = _random_sprite_shape()
        records = _sprite_candidate_records(
            shape_cells,
            marker_rel,
            height,
            width,
            sep_col,
            extra_rows,
            axis,
            target_axis_values,
        )
        if not records:
            return None
        records_by_shape.append(records)

    chosen_records: list[dict]
    if sprite_count == 1:
        chosen_records = [random.choice(records_by_shape[0])]
    else:
        pairs = []
        for rec_a in records_by_shape[0]:
            for rec_b in records_by_shape[1]:
                if not _disjoint(rec_a["coords_in"], rec_b["coords_in"]):
                    continue
                if rec_a["sort_key"] == rec_b["sort_key"]:
                    continue
                overlap = not set(rec_a["final_coords"]).isdisjoint(set(rec_b["final_coords"]))
                if want_overlap and not overlap:
                    continue
                if not want_overlap and overlap:
                    continue
                pairs.append((rec_a, rec_b))
        if not pairs:
            return None
        chosen_records = list(random.choice(pairs))

    body_colors = random.sample(SPRITE_BODY_COLORS, sprite_count)
    raw_sprites = [_make_sprite(record, body_color) for record, body_color in zip(chosen_records, body_colors)]
    ordered_sprites = _sort_sprites_for_side(raw_sprites, anchor_side)

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = SEP_COLOR
    _paint_panel(grid, panel_top, 0, left_mask, LEFT_COLOR)
    _paint_panel(grid, panel_top, sep_col + 1, right_mask, RIGHT_COLOR)

    for sprite in ordered_sprites:
        for (r, c), color in zip(sprite.coords_in, sprite.colors):
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = int(color)

    slots = _anchor_slots(grid, anchor_side, sprite_count)
    if slots is None:
        return None
    slots = [random.choice(slots)] if sprite_count == 1 else list(slots)
    if sprite_count == 2:
        slots = sorted(slots, key=lambda rc: (rc[1], rc[0]) if anchor_side in {"top", "bottom"} else (rc[0], rc[1]))

    anchors = [_anchor_from_cell(r, c, height, width) for r, c in slots]
    for anchor in anchors:
        grid[anchor.r, anchor.c] = MARKER_COLOR

    expected = _build_expected_output(height, panel_w, panel_top, left_mask, right_mask, ordered_sprites, anchors)

    target = (~left_mask) & (~right_mask)
    target_coords = np.argwhere(target)
    if target_coords.shape[0] < 3:
        return None
    if len(set(map(int, target_coords[:, 0]))) == 1 or len(set(map(int, target_coords[:, 1]))) == 1:
        return None
    if not any(sprite.dr != 0 or sprite.dc != 0 for sprite in ordered_sprites):
        return None

    # Semantic validation: each sprite must depend on a unique nearest target axis value.
    for sprite in ordered_sprites:
        source_axis = sprite.marker_out[1] if axis == "h" else sprite.marker_out[0]
        target_axis = _nearest_axis_value(target_axis_values, source_axis)
        if target_axis is None:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    axis_sides = ("top", "bottom") if random.random() < 0.5 else ("left", "right")
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random.choice(axis_sides))
        if grid is not None:
            return grid
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
