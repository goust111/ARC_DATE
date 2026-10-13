from __future__ import annotations

import numpy as np


_BG = 0
_SIDES = ("top", "bottom", "left", "right")
_MAX_GENERATION_ATTEMPTS = 400
_MAX_PLACEMENT_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _open_frame_info(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if len(missing) != 1:
        return None

    mr, mc = map(int, missing[0])
    sides: list[str] = []
    if mr == 0:
        sides.append("top")
    if mr == h - 1:
        sides.append("bottom")
    if mc == 0:
        sides.append("left")
    if mc == w - 1:
        sides.append("right")
    if len(sides) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (r0 + mr, c0 + mc),
        "side": sides[0],
    }


def _line_info(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    length = max(h, w)
    if len(coords) != length:
        return None
    if h == 1 and w == 1:
        axis = "both"
    elif h == 1:
        axis = "h"
    elif w == 1:
        axis = "v"
    else:
        return None
    return {"bbox": (r0, c0, r1, c1), "axis": axis, "length": int(length)}


def _expected_axis(side: str) -> str:
    return "h" if side in {"top", "bottom"} else "v"


def _draw_interior_segment(
    out: np.ndarray,
    color: int,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    gap: tuple[int, int],
    length: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    gap_r, gap_c = gap
    half = length // 2

    if side == "top":
        row = r1 - 1
        start = gap_c - half
        end = start + length
        assert c0 + 1 <= start and end - 1 <= c1 - 1
        out[row, start:end] = color
    elif side == "bottom":
        row = r0 + 1
        start = gap_c - half
        end = start + length
        assert c0 + 1 <= start and end - 1 <= c1 - 1
        out[row, start:end] = color
    elif side == "left":
        col = c1 - 1
        start = gap_r - half
        end = start + length
        assert r0 + 1 <= start and end - 1 <= r1 - 1
        out[start:end, col] = color
    else:
        col = c0 + 1
        start = gap_r - half
        end = start + length
        assert r0 + 1 <= start and end - 1 <= r1 - 1
        out[start:end, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored line sprite into its matching open envelope and seal the gap.

    Input structure:
    - The background is black.
    - Several different colors each appear in exactly two disconnected monochrome objects:
      one is a 1-pixel-thick rectangular frame with a single missing border cell, and the
      other is a 4-connected horizontal or vertical line segment of the same color.
    - Every frame interior is otherwise empty, and different envelopes/sprites do not touch.

    Rule:
    - Pair each line sprite with the same-colored open frame.
    - Locate the unique gap on the frame border; its side determines the inward slide
      direction, and the sprite's long axis stays parallel to that side.
    - Remove the outside sprite, place it inside the frame centered on the gap position,
      and slide it all the way to the opposite inner wall.
    - Repaint the missing border cell with the frame color so the envelope becomes closed.

    Key insight:
    Color identifies which sprite belongs to which frame, while the border gap encodes both
    the insertion direction and the centered final position of the line segment.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        frame_info = None
        sprite_info = None
        sprite_coords = None

        for comp in comps:
            candidate_frame = _open_frame_info(comp)
            if candidate_frame is not None:
                assert frame_info is None, "expected exactly one frame per color"
                frame_info = candidate_frame
                continue

            candidate_line = _line_info(comp)
            if candidate_line is not None:
                assert sprite_info is None, "expected exactly one line sprite per color"
                sprite_info = candidate_line
                sprite_coords = comp

        assert frame_info is not None, "missing open frame"
        assert sprite_info is not None and sprite_coords is not None, "missing line sprite"

        side = str(frame_info["side"])
        axis = str(sprite_info["axis"])
        assert axis in {"both", _expected_axis(side)}, "sprite orientation does not match gap side"

        out[sprite_coords[:, 0], sprite_coords[:, 1]] = bg
        gap = tuple(int(v) for v in frame_info["gap"])
        out[gap] = color
        _draw_interior_segment(
            out=out,
            color=color,
            frame_bbox=tuple(int(v) for v in frame_info["bbox"]),
            side=side,
            gap=gap,
            length=int(sprite_info["length"]),
        )

    return out


def _padded_slice(
    grid_shape: tuple[int, int],
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    gh, gw = grid_shape
    return (
        max(0, int(r0 - pad)),
        max(0, int(c0 - pad)),
        min(int(gh), int(r0 + h + pad)),
        min(int(gw), int(c0 + w + pad)),
    )


def _region_is_free(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int = 1,
) -> bool:
    rr0, cc0, rr1, cc1 = _padded_slice(occupied.shape, r0, c0, h, w, pad=pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve_region(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int = 1,
) -> None:
    rr0, cc0, rr1, cc1 = _padded_slice(occupied.shape, r0, c0, h, w, pad=pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_open_frame(
    grid: np.ndarray,
    color: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
    gap_offset: int,
) -> tuple[int, int]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        gap = (r0, c0 + gap_offset)
    elif side == "bottom":
        gap = (r1, c0 + gap_offset)
    elif side == "left":
        gap = (r0 + gap_offset, c0)
    else:
        gap = (r0 + gap_offset, c1)

    grid[gap] = _BG
    return int(gap[0]), int(gap[1])


def _draw_closed_frame(grid: np.ndarray, color: int, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sprite_patch(color: int, side: str, length: int) -> np.ndarray:
    if side in {"top", "bottom"}:
        return np.full((1, length), color, dtype=int)
    return np.full((length, 1), color, dtype=int)


def _sample_frame_spec(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
    color: int,
) -> dict[str, object] | None:
    gh, gw = grid_shape
    max_h = min(10, gh - 2)
    max_w = min(10, gw - 2)
    if max_h < 5 or max_w < 5:
        return None

    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        h = int(rng.integers(5, max_h + 1))
        w = int(rng.integers(5, max_w + 1))
        side = str(rng.choice(_SIDES))
        span = (w - 2) if side in {"top", "bottom"} else (h - 2)
        valid_lengths = [length for length in range(1, span + 1) if length % 2 == 1]
        if not valid_lengths:
            continue
        length = int(rng.choice(valid_lengths))
        half = length // 2
        offset_lo = 1 + half
        offset_hi = (w - 2 - half) if side in {"top", "bottom"} else (h - 2 - half)
        if offset_lo > offset_hi:
            continue
        gap_offset = int(rng.integers(offset_lo, offset_hi + 1))
        r0 = int(rng.integers(1, gh - h))
        c0 = int(rng.integers(1, gw - w))
        if not _region_is_free(occupied, r0, c0, h, w, pad=1):
            continue
        _reserve_region(occupied, r0, c0, h, w, pad=1)
        return {
            "color": color,
            "r0": r0,
            "c0": c0,
            "h": h,
            "w": w,
            "side": side,
            "length": length,
            "gap_offset": gap_offset,
        }
    return None


def _place_sprite(
    rng: np.random.Generator,
    occupied: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    gh, gw = occupied.shape
    if ph > gh or pw > gw:
        return None

    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        r0 = int(rng.integers(0, gh - ph + 1))
        c0 = int(rng.integers(0, gw - pw + 1))
        if not _region_is_free(occupied, r0, c0, ph, pw, pad=1):
            continue
        _reserve_region(occupied, r0, c0, ph, pw, pad=1)
        return r0, c0
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(2, 5))
    min_side = 16 + 2 * n_pairs
    grid_h = int(rng.integers(min_side, 31))
    grid_w = int(rng.integers(min_side, 31))
    colors = [int(v) for v in rng.choice(np.arange(1, 10), size=n_pairs, replace=False)]

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    specs: list[dict[str, object]] = []
    for color in colors:
        spec = _sample_frame_spec(rng, (grid_h, grid_w), occupied, color)
        if spec is None:
            return None
        specs.append(spec)

    specs_sorted = sorted(specs, key=lambda spec: int(spec["length"]), reverse=True)
    for spec in specs_sorted:
        patch = _sprite_patch(int(spec["color"]), str(spec["side"]), int(spec["length"]))
        loc = _place_sprite(rng, occupied, patch)
        if loc is None:
            return None
        spec["sprite_r0"], spec["sprite_c0"] = loc

    inp = np.full((grid_h, grid_w), _BG, dtype=int)
    out = np.full((grid_h, grid_w), _BG, dtype=int)

    for spec in specs:
        color = int(spec["color"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        side = str(spec["side"])
        length = int(spec["length"])
        gap_offset = int(spec["gap_offset"])
        sprite_r0 = int(spec["sprite_r0"])
        sprite_c0 = int(spec["sprite_c0"])

        gap = _draw_open_frame(inp, color, r0, c0, h, w, side, gap_offset)
        _draw_closed_frame(out, color, r0, c0, h, w)
        _draw_interior_segment(out, color, (r0, c0, r0 + h - 1, c0 + w - 1), side, gap, length)

        patch = _sprite_patch(color, side, length)
        ph, pw = patch.shape
        inp[sprite_r0 : sprite_r0 + ph, sprite_c0 : sprite_c0 + pw] = patch

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_GENERATION_ATTEMPTS} attempts")
