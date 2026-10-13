from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MASTER_SIZE = 30
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _mark_rectangle_core(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    core = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(mask[r : r + 2, c : c + 2].all()):
                core[r : r + 2, c : c + 2] = True
    return core


def _extract_objects(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    object_pixels = np.zeros_like(g, dtype=bool)
    objects_by_color: dict[int, dict[str, int]] = {}
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        mask = g == color
        core = _mark_rectangle_core(mask)
        assert bool(core.any()), "each object rectangle must be at least 2x2"
        rows, cols = np.where(core)
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        rect = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(rect.all()), "rectangle bbox must be filled by its color"
        rect_mask = np.zeros_like(mask, dtype=bool)
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        object_pixels |= rect_mask
        objects_by_color[color] = {
            "color": color,
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
        }
    return objects_by_color, object_pixels


def _apply_transfer(
    out: np.ndarray,
    rect: dict[str, int],
    source_color: int,
    amount: int,
    target_face: str,
) -> None:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    if target_face in ("left", "right"):
        width = c1 - c0 + 1
        take = min(int(amount), int(width))
        if target_face == "left":
            out[r0 : r1 + 1, c0 : c0 + take] = int(source_color)
        else:
            out[r0 : r1 + 1, c1 - take + 1 : c1 + 1] = int(source_color)
    else:
        assert target_face in ("top", "bottom")
        height = r1 - r0 + 1
        take = min(int(amount), int(height))
        if target_face == "top":
            out[r0 : r0 + take, c0 : c1 + 1] = int(source_color)
        else:
            out[r1 - take + 1 : r1 + 1, c0 : c1 + 1] = int(source_color)


def _parse_bridge(
    grid: np.ndarray,
    bridge: list[tuple[int, int]],
    object_pixels: np.ndarray,
) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bridge = sorted((int(r), int(c)) for r, c in bridge)
    color = int(g[bridge[0]])
    assert all(int(g[r, c]) == color for r, c in bridge), "bridge must be monochrome"

    rows = {r for r, _c in bridge}
    cols = {c for _r, c in bridge}
    horizontal = len(rows) == 1
    vertical = len(cols) == 1
    assert horizontal ^ vertical, "bridge must be straight"

    if horizontal:
        r = next(iter(rows))
        c0 = min(c for _r, c in bridge)
        c1 = max(c for _r, c in bridge)
        assert len(bridge) == c1 - c0 + 1, "horizontal bridge must be contiguous"
        left_obj = (
            (r, c0 - 1)
            if c0 > 0 and object_pixels[r, c0 - 1]
            else None
        )
        right_obj = (
            (r, c1 + 1)
            if c1 + 1 < g.shape[1] and object_pixels[r, c1 + 1]
            else None
        )
        assert left_obj is not None and right_obj is not None, "bridge must touch objects at both ends"
        left_color = int(g[left_obj])
        right_color = int(g[right_obj])
        assert left_color != right_color, "connected objects must have different colors"
        if left_color == color:
            source_color = left_color
            target_color = right_color
            target_face = "left"
        else:
            assert right_color == color, "bridge color must match one endpoint object"
            source_color = right_color
            target_color = left_color
            target_face = "right"
        return {
            "orientation": "horizontal",
            "length": len(bridge),
            "source_color": source_color,
            "target_color": target_color,
            "target_face": target_face,
            "anchor": (r, c0),
        }

    c = next(iter(cols))
    r0 = min(r for r, _c in bridge)
    r1 = max(r for r, _c in bridge)
    assert len(bridge) == r1 - r0 + 1, "vertical bridge must be contiguous"
    top_obj = (
        (r0 - 1, c)
        if r0 > 0 and object_pixels[r0 - 1, c]
        else None
    )
    bottom_obj = (
        (r1 + 1, c)
        if r1 + 1 < g.shape[0] and object_pixels[r1 + 1, c]
        else None
    )
    assert top_obj is not None and bottom_obj is not None, "bridge must touch objects at both ends"
    top_color = int(g[top_obj])
    bottom_color = int(g[bottom_obj])
    assert top_color != bottom_color, "connected objects must have different colors"
    if top_color == color:
        source_color = top_color
        target_color = bottom_color
        target_face = "top"
    else:
        assert bottom_color == color, "bridge color must match one endpoint object"
        source_color = bottom_color
        target_color = top_color
        target_face = "bottom"
    return {
        "orientation": "vertical",
        "length": len(bridge),
        "source_color": source_color,
        "target_color": target_color,
        "target_face": target_face,
        "anchor": (r0, c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer a source rectangle's color through each bridge into edge-aligned stripes on the opposite rectangle.

    Input structure:
    - The background is Black.
    - Several disconnected monochrome solid rectangles of distinct colors appear on the grid.
    - Some rectangle pairs are connected by a single straight 1-pixel-wide horizontal or vertical bridge.
    - Each bridge has the same color as exactly one of the two rectangles it connects.

    Rule:
    - For every bridge, identify the endpoint rectangle whose color matches the bridge; that rectangle is the source and the other endpoint rectangle is the target.
    - Let N be the bridge length in pixels.
    - If the bridge is horizontal, recolor the target's N nearest columns from the side touching the bridge to the source color.
    - If the bridge is vertical, recolor the target's N nearest rows from the side touching the bridge to the source color.
    - If the target is narrower or shorter than N along that direction, recolor the whole target rectangle.
    - Leave every bridge pixel and every source rectangle unchanged.

    Key insight:
    - A bridge simultaneously encodes the paired objects, the transfer direction through its matching color, and the transfer amount through its length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    objects_by_color, object_pixels = _extract_objects(g, bg)
    bridge_mask = (g != bg) & ~object_pixels
    bridges = _components4(bridge_mask)
    assert bridges, "expected at least one bridge"

    parsed = [
        _parse_bridge(g, bridge, object_pixels)
        for bridge in bridges
    ]
    target_colors = [int(item["target_color"]) for item in parsed]
    assert len(target_colors) == len(set(target_colors)), "generator assumes each target has at most one bridge"

    out = g.copy()
    for item in sorted(parsed, key=lambda x: tuple(x["anchor"])):
        target_color = int(item["target_color"])
        rect = objects_by_color[target_color]
        _apply_transfer(
            out,
            rect,
            int(item["source_color"]),
            int(item["length"]),
            str(item["target_face"]),
        )
    return out


def _balanced_bools(count: int) -> list[bool]:
    values = [bool(random.getrandbits(1)) for _ in range(count)]
    if count >= 2 and (all(values) or not any(values)):
        values[0] = False
        values[1] = True
        random.shuffle(values)
    return values


def _balanced_orientations(count: int) -> list[str]:
    values = [random.choice(("horizontal", "vertical")) for _ in range(count)]
    if count >= 2 and len(set(values)) == 1:
        values[0] = "horizontal"
        values[1] = "vertical"
        random.shuffle(values)
    return values


def _sample_pair_patch(
    source_color: int,
    target_color: int,
    orientation: str,
    source_first: bool,
    force_full_cover: bool,
) -> dict[str, object] | None:
    for _ in range(200):
        if force_full_cover:
            bridge_len = random.randint(3, 5)
        else:
            bridge_len = random.randint(1, 5)

        hs = random.randint(2, 6)
        ws = random.randint(2, 6)
        ht = random.randint(2, 6)
        wt = random.randint(2, 6)

        if orientation == "horizontal":
            if force_full_cover:
                if bridge_len <= 2:
                    continue
                wt = random.randint(2, bridge_len - 1)
            else:
                wt = random.randint(max(2, bridge_len), 6)

            src_row = random.randint(0, hs - 1)
            tgt_row = random.randint(0, ht - 1)
            joint_row = max(src_row, tgt_row)
            s_top = joint_row - src_row
            t_top = joint_row - tgt_row
            row = joint_row
            height = max(s_top + hs, t_top + ht)
            width = ws + bridge_len + wt
            patch = np.zeros((height, width), dtype=int)

            if source_first:
                s_left = 0
                bridge_start = ws
                t_left = ws + bridge_len
                target_face = "left"
            else:
                t_left = 0
                bridge_start = wt
                s_left = wt + bridge_len
                target_face = "right"

            patch[s_top : s_top + hs, s_left : s_left + ws] = source_color
            patch[t_top : t_top + ht, t_left : t_left + wt] = target_color
            patch[row, bridge_start : bridge_start + bridge_len] = source_color

            return {
                "patch": patch,
                "source_rect": (s_top, s_left, s_top + hs - 1, s_left + ws - 1),
                "target_rect": (t_top, t_left, t_top + ht - 1, t_left + wt - 1),
                "source_color": source_color,
                "target_color": target_color,
                "length": bridge_len,
                "target_face": target_face,
            }

        if force_full_cover:
            if bridge_len <= 2:
                continue
            ht = random.randint(2, bridge_len - 1)
        else:
            ht = random.randint(max(2, bridge_len), 6)

        src_col = random.randint(0, ws - 1)
        tgt_col = random.randint(0, wt - 1)
        joint_col = max(src_col, tgt_col)
        s_left = joint_col - src_col
        t_left = joint_col - tgt_col
        col = joint_col
        height = hs + bridge_len + ht
        width = max(s_left + ws, t_left + wt)
        patch = np.zeros((height, width), dtype=int)

        if source_first:
            s_top = 0
            bridge_start = hs
            t_top = hs + bridge_len
            target_face = "top"
        else:
            t_top = 0
            bridge_start = ht
            s_top = ht + bridge_len
            target_face = "bottom"

        patch[s_top : s_top + hs, s_left : s_left + ws] = source_color
        patch[t_top : t_top + ht, t_left : t_left + wt] = target_color
        patch[bridge_start : bridge_start + bridge_len, col] = source_color

        return {
            "patch": patch,
            "source_rect": (s_top, s_left, s_top + hs - 1, s_left + ws - 1),
            "target_rect": (t_top, t_left, t_top + ht - 1, t_left + wt - 1),
            "source_color": source_color,
            "target_color": target_color,
            "length": bridge_len,
            "target_face": target_face,
        }
    return None


def _sample_isolated_patch(color: int) -> np.ndarray:
    height = random.randint(2, 6)
    width = random.randint(2, 6)
    patch = np.zeros((height, width), dtype=int)
    patch[:, :] = color
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_patch(
    canvas: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    mask = patch != BG
    blocked = _dilate8(occupied)
    candidates: list[tuple[int, int]] = []

    for top in range(canvas.shape[0] - ph + 1):
        for left in range(canvas.shape[1] - pw + 1):
            if bool(np.any(blocked[top : top + ph, left : left + pw] & mask)):
                continue
            candidates.append((top, left))

    if not candidates:
        return None

    top, left = random.choice(candidates)
    region = canvas[top : top + ph, left : left + pw]
    assert not bool(np.any((region != BG) & mask))
    region[mask] = patch[mask]
    occupied[top : top + ph, left : left + pw] |= mask
    return top, left


def _shift_rect(rect: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _crop_canvas(
    canvas: np.ndarray,
    pair_specs: list[dict[str, object]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    coords = np.argwhere(canvas != BG)
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())

    top_margin = random.randint(0, min(4, r0))
    left_margin = random.randint(0, min(4, c0))
    bottom_margin = random.randint(0, min(4, canvas.shape[0] - 1 - r1))
    right_margin = random.randint(0, min(4, canvas.shape[1] - 1 - c1))

    rr0 = r0 - top_margin
    rr1 = r1 + bottom_margin
    cc0 = c0 - left_margin
    cc1 = c1 + right_margin
    cropped = canvas[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()

    shifted_specs: list[dict[str, object]] = []
    for spec in pair_specs:
        shifted_specs.append(
            {
                "source_rect": _shift_rect(spec["source_rect"], -rr0, -cc0),
                "target_rect": _shift_rect(spec["target_rect"], -rr0, -cc0),
                "source_color": int(spec["source_color"]),
                "target_color": int(spec["target_color"]),
                "length": int(spec["length"]),
                "target_face": str(spec["target_face"]),
            }
        )
    return cropped, shifted_specs


def _oracle_output(grid: np.ndarray, pair_specs: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in pair_specs:
        rect = spec["target_rect"]
        rect_info = {
            "r0": int(rect[0]),
            "c0": int(rect[1]),
            "r1": int(rect[2]),
            "c1": int(rect[3]),
        }
        _apply_transfer(
            out,
            rect_info,
            int(spec["source_color"]),
            int(spec["length"]),
            str(spec["target_face"]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    extra_count = random.randint(0, min(2, 9 - 2 * pair_count))
    colors = random.sample(PALETTE, 2 * pair_count + extra_count)

    orientations = _balanced_orientations(pair_count)
    source_first_flags = _balanced_bools(pair_count)
    full_cover_flags = _balanced_bools(pair_count)

    canvas = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    placed_pairs: list[dict[str, object]] = []

    color_idx = 0
    for i in range(pair_count):
        source_color = colors[color_idx]
        target_color = colors[color_idx + 1]
        color_idx += 2
        spec = _sample_pair_patch(
            source_color,
            target_color,
            orientations[i],
            source_first_flags[i],
            full_cover_flags[i],
        )
        if spec is None:
            return None
        patch = np.asarray(spec["patch"], dtype=int)
        placement = _place_patch(canvas, occupied, patch)
        if placement is None:
            return None
        top, left = placement
        placed_pairs.append(
            {
                "source_rect": _shift_rect(spec["source_rect"], top, left),
                "target_rect": _shift_rect(spec["target_rect"], top, left),
                "source_color": int(source_color),
                "target_color": int(target_color),
                "length": int(spec["length"]),
                "target_face": str(spec["target_face"]),
            }
        )

    while color_idx < len(colors):
        patch = _sample_isolated_patch(colors[color_idx])
        color_idx += 1
        placement = _place_patch(canvas, occupied, patch)
        if placement is None:
            return None

    cropped, shifted_specs = _crop_canvas(canvas, placed_pairs)
    oracle = _oracle_output(cropped, shifted_specs)

    try:
        produced = main(cropped)
    except AssertionError:
        return None

    if not np.array_equal(produced, oracle):
        return None
    if np.array_equal(cropped, oracle):
        return None
    if cropped.shape[0] > 30 or cropped.shape[1] > 30:
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-transfer sample")
