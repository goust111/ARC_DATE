import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BG = 0
GREY = 5
ANCHOR_SIZE = 5
SLOT_OFFSETS = {
    "cross": (1, 1),
    "top_border": (1, 3),
    "side_border": (3, 1),
    "periodic": (3, 3),
}
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _filled_rect(comp: dict) -> bool:
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    return int(comp["area"]) == h * w


def _solid_color_patch(color: int) -> np.ndarray:
    return np.full((ANCHOR_SIZE, ANCHOR_SIZE), int(color), dtype=int)


def _decode_slots(anchor_patch: np.ndarray) -> dict[str, int]:
    return {
        name: int(anchor_patch[r, c])
        for name, (r, c) in SLOT_OFFSETS.items()
    }


def _looks_like_anchor_patch(patch: np.ndarray) -> bool:
    if patch.shape != (ANCHOR_SIZE, ANCHOR_SIZE):
        return False
    for r in range(ANCHOR_SIZE):
        for c in range(ANCHOR_SIZE):
            if (r, c) in SLOT_OFFSETS.values():
                continue
            if int(patch[r, c]) != GREY:
                return False
    return True


def _structure(grid: np.ndarray):
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False)
    rects = [comp for comp in comps if _filled_rect(comp)]

    source = None
    dest = None
    for comp in rects:
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if (h, w) != (ANCHOR_SIZE, ANCHOR_SIZE):
            continue
        patch = crop_bbox(grid, comp["bbox"])
        if _looks_like_anchor_patch(patch) and np.any(patch != GREY):
            if source is None or comp["bbox"] < source["bbox"]:
                source = comp
        elif np.all(patch == GREY):
            if dest is None or comp["bbox"] > dest["bbox"]:
                dest = comp

    assert source is not None, "missing source anchor"
    assert dest is not None, "missing diagonal grey rectangle"

    sr0, sc0, sr1, sc1 = source["bbox"]
    dr0, dc0, dr1, dc1 = dest["bbox"]
    assert sr0 < dr0 and sc0 < dc0, "anchors must be on opposite diagonal corners"

    vertical_bars = []
    horizontal_bars = []
    for comp in rects:
        if comp is source or comp is dest:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        color = int(crop_bbox(grid, comp["bbox"])[0, 0])
        if (
            bbox_h(comp["bbox"]) == ANCHOR_SIZE
            and bbox_w(comp["bbox"]) == 1
            and r0 == sr0
            and c0 > sc1
            and c0 < dc0
        ):
            vertical_bars.append((c0, color))
        elif (
            bbox_h(comp["bbox"]) == 1
            and bbox_w(comp["bbox"]) == ANCHOR_SIZE
            and c0 == sc0
            and r0 > sr1
            and r0 < dr0
        ):
            horizontal_bars.append((r0, color))

    vertical_bars.sort()
    horizontal_bars.sort()
    assert len(vertical_bars) == 2, "expected two vertical control bars"
    assert len(horizontal_bars) == 2, "expected two horizontal control bars"

    return {
        "source_bbox": source["bbox"],
        "dest_bbox": dest["bbox"],
        "source_patch": crop_bbox(grid, source["bbox"]),
        "vertical_bars": vertical_bars,
        "horizontal_bars": horizontal_bars,
    }


def _periodic_positions(lo: int, hi: int, phase: int, period: int) -> list[int]:
    positions = []
    k0 = int(np.floor((lo - phase) / period))
    k1 = int(np.ceil((hi - phase) / period))
    for k in range(k0, k1 + 1):
        pos = phase + k * period
        if lo <= pos <= hi:
            positions.append(int(pos))
    return positions


def _build_output(
    shape: tuple[int, int],
    source_bbox: tuple[int, int, int, int],
    dest_bbox: tuple[int, int, int, int],
    source_patch: np.ndarray,
    slots: dict[str, int],
    phase_col: int,
    period_h: int,
    phase_row: int,
    period_v: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    sr0, sc0, sr1, sc1 = source_bbox
    dr0, dc0, dr1, dc1 = dest_bbox

    paste(out, source_patch, sr0, sc0)
    paste(out, source_patch, sr0, dc0)
    paste(out, source_patch, dr0, sc0)
    paste(out, _solid_color_patch(GREY), dr0, dc0)

    top = sr0
    bottom = dr1
    left = sc0
    right = dc1

    out[top, left : right + 1] = int(slots["top_border"])
    out[bottom, left : right + 1] = int(slots["top_border"])
    out[top : bottom + 1, left] = int(slots["side_border"])
    out[top : bottom + 1, right] = int(slots["side_border"])

    periodic_cols = _periodic_positions(left + 1, right - 1, phase_col, period_h)
    periodic_rows = _periodic_positions(top + 1, bottom - 1, phase_row, period_v)

    top_rows = np.arange(top + 1, sr1 + 1)
    bottom_rows = np.arange(dr0, bottom)
    left_cols = np.arange(left + 1, sc1 + 1)
    right_cols = np.arange(dc0, right)

    vertical_mask = np.zeros(shape, dtype=bool)
    horizontal_mask = np.zeros(shape, dtype=bool)
    for c in periodic_cols:
        vertical_mask[top_rows, c] = True
        vertical_mask[bottom_rows, c] = True
    for r in periodic_rows:
        horizontal_mask[r, left_cols] = True
        horizontal_mask[r, right_cols] = True

    out[vertical_mask] = int(slots["periodic"])
    out[horizontal_mask] = int(slots["periodic"])
    out[vertical_mask & horizontal_mask] = int(slots["cross"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend inside the anchor block and use the control-bar phase and period to decorate the four-corner frame.

    Input structure:
    - The background is black.
    - A 5x5 grey anchor block sits in the upper-left area; only its four odd/odd
      interior legend slots may differ from grey.
    - A same-sized solid grey block sits on the opposite diagonal corner, while
      two vertical control bars are to the right of the anchor and two horizontal
      control bars are below it.

    Rule:
    - Read the four legend slots: `(1,1)` is the crossing color, `(1,3)` is the
      top/bottom outer-border color, `(3,1)` is the left/right outer-border
      color, and `(3,3)` is the periodic decoration color.
    - Copy the decorated anchor block to the missing top-right and bottom-left
      corners, keep the lower-right block grey, and treat the two bar pairs as
      the first two ticks of the horizontal and vertical periods.
    - Draw the outer rectangle borders with the decoded border colors, repeat the
      periodic decoration on the top/bottom bands and left/right bands using the
      control-bar phase, and recolor periodic crossings with the `(1,1)` legend color.

    Key insight:
    - The anchor's legend slots define which frame layer gets which color, while
      the control bars contribute only the period and phase of the repeated border pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(get_bg(g)) == BG, "background must be black"

    info = _structure(g)
    source_patch = info["source_patch"]
    slots = _decode_slots(source_patch)

    (c1, vc1), (c2, vc2) = info["vertical_bars"]
    (r1, hc1), (r2, hc2) = info["horizontal_bars"]
    period_h = int(c2 - c1)
    period_v = int(r2 - r1)
    assert period_h > 0 and period_v > 0, "control bars must be separated"

    assert int(vc1) == int(slots["top_border"]), "first vertical control bar must match top-border slot"
    assert int(vc2) == int(slots["periodic"]), "second vertical control bar must match periodic slot"
    assert int(hc1) == int(slots["side_border"]), "first horizontal control bar must match side-border slot"
    assert int(hc2) == int(slots["periodic"]), "second horizontal control bar must match periodic slot"

    sr0, sc0, _sr1, _sc1 = info["source_bbox"]
    dr0, dc0, _dr1, _dc1 = info["dest_bbox"]
    assert (dc0 - sc0) % period_h == 0, "horizontal span must be period-aligned"
    assert (dr0 - sr0) % period_v == 0, "vertical span must be period-aligned"

    return _build_output(
        g.shape,
        info["source_bbox"],
        info["dest_bbox"],
        source_patch,
        slots,
        c1,
        period_h,
        r1,
        period_v,
    )


def _make_anchor_patch() -> tuple[np.ndarray, dict[str, int]]:
    patch = _solid_color_patch(GREY)
    grey_slot = random.choice(list(SLOT_OFFSETS))
    colors = random.sample(PALETTE, 3)
    slots = {}
    color_iter = iter(colors)
    for name, (r, c) in SLOT_OFFSETS.items():
        value = GREY if name == grey_slot else next(color_iter)
        slots[name] = int(value)
        patch[r, c] = int(value)
    assert len(set(slots.values())) == 4
    return patch, slots


def _alt_without_cross(
    shape: tuple[int, int],
    source_bbox: tuple[int, int, int, int],
    dest_bbox: tuple[int, int, int, int],
    source_patch: np.ndarray,
    slots: dict[str, int],
    phase_col: int,
    period_h: int,
    phase_row: int,
    period_v: int,
) -> np.ndarray:
    alt_slots = dict(slots)
    alt_slots["cross"] = alt_slots["periodic"]
    return _build_output(
        shape,
        source_bbox,
        dest_bbox,
        source_patch,
        alt_slots,
        phase_col,
        period_h,
        phase_row,
        period_v,
    )


def _alt_without_copies(
    shape: tuple[int, int],
    source_bbox: tuple[int, int, int, int],
    dest_bbox: tuple[int, int, int, int],
    slots: dict[str, int],
    phase_col: int,
    period_h: int,
    phase_row: int,
    period_v: int,
) -> np.ndarray:
    blank = _solid_color_patch(GREY)
    return _build_output(
        shape,
        source_bbox,
        dest_bbox,
        blank,
        slots,
        phase_col,
        period_h,
        phase_row,
        period_v,
    )


def _alt_without_periodic(
    shape: tuple[int, int],
    source_bbox: tuple[int, int, int, int],
    dest_bbox: tuple[int, int, int, int],
    source_patch: np.ndarray,
    slots: dict[str, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    sr0, sc0, _sr1, _sc1 = source_bbox
    dr0, dc0, dr1, dc1 = dest_bbox
    paste(out, source_patch, sr0, sc0)
    paste(out, source_patch, sr0, dc0)
    paste(out, source_patch, dr0, sc0)
    paste(out, _solid_color_patch(GREY), dr0, dc0)
    out[sr0, sc0 : dc1 + 1] = int(slots["top_border"])
    out[dr1, sc0 : dc1 + 1] = int(slots["top_border"])
    out[sr0 : dr1 + 1, sc0] = int(slots["side_border"])
    out[sr0 : dr1 + 1, dc1] = int(slots["side_border"])
    return out


def _visible_source_detail(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    patch: np.ndarray,
) -> bool:
    r0, c0, _r1, _c1 = bbox
    for name, (rr, cc) in SLOT_OFFSETS.items():
        if int(patch[rr, cc]) == GREY:
            continue
        if int(grid[r0 + rr, c0 + cc]) == int(patch[rr, cc]):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    patch, slots = _make_anchor_patch()
    period_h = random.randint(2, 5)
    period_v = random.randint(2, 5)
    k_h = random.randint(4, 5)
    k_v = random.randint(4, 5)
    struct_w = ANCHOR_SIZE + k_h * period_h
    struct_h = ANCHOR_SIZE + k_v * period_v
    if struct_w > 26 or struct_h > 26:
        return None

    rows = random.randint(struct_h + 2, min(30, struct_h + 5))
    cols = random.randint(struct_w + 2, min(30, struct_w + 5))
    top_margin = random.randint(1, rows - struct_h - 1)
    left_margin = random.randint(1, cols - struct_w - 1)

    sr0 = top_margin
    sc0 = left_margin
    dr0 = sr0 + k_v * period_v
    dc0 = sc0 + k_h * period_h
    sr1 = sr0 + ANCHOR_SIZE - 1
    sc1 = sc0 + ANCHOR_SIZE - 1
    dr1 = dr0 + ANCHOR_SIZE - 1
    dc1 = dc0 + ANCHOR_SIZE - 1

    c1 = sc1 + 2
    c2 = c1 + period_h
    r1 = sr1 + 2
    r2 = r1 + period_v
    if not (c2 < dc0 and r2 < dr0):
        return None

    inp = np.full((rows, cols), BG, dtype=int)
    paste(inp, patch, sr0, sc0)
    paste(inp, _solid_color_patch(GREY), dr0, dc0)
    inp[sr0 : sr1 + 1, c1] = int(slots["top_border"])
    inp[sr0 : sr1 + 1, c2] = int(slots["periodic"])
    inp[r1, sc0 : sc1 + 1] = int(slots["side_border"])
    inp[r2, sc0 : sc1 + 1] = int(slots["periodic"])

    expected = _build_output(
        inp.shape,
        (sr0, sc0, sr1, sc1),
        (dr0, dc0, dr1, dc1),
        patch,
        slots,
        c1,
        period_h,
        r1,
        period_v,
    )

    if not _visible_source_detail(expected, (sr0, sc0, sr1, sc1), patch):
        return None
    if not _visible_source_detail(expected, (sr0, dc0, sr1, dc1), patch):
        return None
    if not _visible_source_detail(expected, (dr0, sc0, dr1, sc1), patch):
        return None

    if np.array_equal(expected, _alt_without_cross(inp.shape, (sr0, sc0, sr1, sc1), (dr0, dc0, dr1, dc1), patch, slots, c1, period_h, r1, period_v)):
        return None
    if np.array_equal(expected, _alt_without_copies(inp.shape, (sr0, sc0, sr1, sc1), (dr0, dc0, dr1, dc1), slots, c1, period_h, r1, period_v)):
        return None
    if np.array_equal(expected, _alt_without_periodic(inp.shape, (sr0, sc0, sr1, sc1), (dr0, dc0, dr1, dc1), patch, slots)):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
