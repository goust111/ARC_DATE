# Variant of 7ed72f31
# Dimension: D13 — 对称性修复替代镜像生成 (transform)
# Change: 在背景上生成全新的镜像副本（添加像素）... → 对称性修复：删除破坏对称的像素
#
# Design analysis
# a) Concrete behavior:
#    The original task shows red as a row-axis / col-axis / point-center and asks us
#    to add a missing mirrored copy onto background. For D13 we keep the same marker
#    language, but the colored object is already almost symmetric around that marker.
#    A few extra payload pixels appear only on one side. The output deletes exactly
#    those unmatched pixels, leaving a perfectly symmetric sprite.
#    Example: a horizontal red line with a green payload on both sides used to mean
#    "copy the top payload to the bottom". In this variant the bottom is already
#    present; input adds one extra green cell on the top whose reflected location is
#    background. Old logic would add a bottom cell. New logic instead removes the top
#    extra cell and keeps all already-matched pairs.
# b) Complexity calibration:
#    To make the inversion visible, every generated group is fully symmetric in the
#    target output and then corrupted by 1-2 one-sided extra pixels in the input.
#    We keep only three marker types (row / col / isolated point) and one payload
#    color per group. This is enough variety to force inference, while avoiding
#    ambiguous multi-color matching inside one symmetry group.
# c) Implicit-assumption audit of the original solver:
#    - The original component scan is fine: red geometry still identifies axis type.
#    - The original "eligible(...)" + write mirrored pixel assumes the reflected cell
#      should be painted, so scanning all grid pixels is harmless. That becomes wrong
#      here because unrelated pixels would be at risk of deletion. We therefore only
#      inspect payload pixels that belong to the marker's own component (row/col) or
#      to the single 8-connected component adjacent to an isolated center.
#    - The original unconditional assignment `out[rr, cc] = v` relies on "adding onto
#      background". In D13 the safe action is the opposite: keep a payload pixel only
#      if its reflected counterpart already exists with the same color.
# d) Placement legality:
#    Generator builds each group as a local patch first, including the future extra
#    pixels. Patches are placed with a one-cell moat so groups never touch, never go
#    out of bounds, and never create accidental alternative components. Isolated-point
#    groups avoid 4-neighbors of the red center so the center stays isolated.
# e) Visual observability:
#    Each sample contains row, col, and isolated-point cases. The input/output change
#    is always a visible deletion (never zero-change). Since the red marker remains in
#    place and the surviving payload is clearly symmetric afterwards, a human can infer
#    "remove unmatched pixels" from 2-3 examples.
# f) Visual cleanliness:
#    No free distractor sprites are generated. Each group uses one color, sits in its
#    own padded region, and has a compact footprint around its red marker.
# g) Color reasoning:
#    Colors are not remapped. Color only binds pixels into a single sprite; symmetry is
#    the only transformation signal. This keeps the variant about geometry, not color.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask

MARK = 2
LAST_META: dict[str, object] = {}


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - int(r)), int(2 * pc - int(c))


def _repair_pixels(
    g: np.ndarray,
    out: np.ndarray,
    pixels: list[tuple[int, int]],
    reflect_fn,
    bg: int,
) -> None:
    keep: set[tuple[int, int]] = set()
    for r, c in pixels:
        v = int(g[r, c])
        rr, cc = reflect_fn(int(r), int(c))
        if _inb(g.shape, rr, cc) and int(g[rr, cc]) == v:
            keep.add((int(r), int(c)))
    for r, c in pixels:
        if (int(r), int(c)) not in keep:
            out[int(r), int(c)] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair near-symmetry around red markers by deleting unmatched payload pixels.

    Marker geometry is unchanged from the source task:
    - horizontal red line => reflection across that row
    - vertical red line   => reflection across that column
    - isolated red pixel  => point symmetry around that center
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    lab4, n4 = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n4) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    # Mixed components: row-axis / col-axis / embedded point-center.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue
        if len(coords) == 1 and len(reds) == 1:
            continue  # handled later as isolated marker

        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        payload = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not payload:
            continue

        if len(reds) == 1:
            center = (int(reds[0][0]), int(reds[0][1]))
            _repair_pixels(g, out, payload, lambda r, c, p=center: _refl_point(p, r, c), bg)
        elif len(rs) == 1:
            r0 = int(next(iter(rs)))
            _repair_pixels(g, out, payload, lambda r, c, rr=r0: _refl_row(rr, r, c), bg)
        elif len(cs) == 1:
            c0 = int(next(iter(cs)))
            _repair_pixels(g, out, payload, lambda r, c, cc=c0: _refl_col(cc, r, c), bg)

    # Isolated red center: repair the single adjacent 8-connected payload component.
    payload_mask = (g != int(bg)) & (g != int(MARK))
    lab8, _n8 = label_mask(payload_mask, connectivity=8)
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue
        pr, pc = map(int, coords[0])
        seen_labels: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(g.shape, rr, cc):
                    continue
                lbl = int(lab8[rr, cc])
                if lbl != 0:
                    seen_labels.add(lbl)
        for lbl in seen_labels:
            pixels = [(int(r), int(c)) for r, c in np.argwhere(lab8 == int(lbl))]
            _repair_pixels(g, out, pixels, lambda r, c, p=(pr, pc): _refl_point(p, r, c), bg)

    return out


def _connected_shape(rng: random.Random, h: int, w: int, count: int) -> set[tuple[int, int]]:
    while True:
        cells = {(int(h - 1), int(rng.randrange(w)))}
        while len(cells) < int(count):
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) == int(count):
            return cells


def _build_row_patch(rng: random.Random, bg: int, color: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    while True:
        half_h = rng.randint(2, 4)
        width = rng.randint(3, 6)
        area = int(half_h * width)
        base_n = rng.randint(3, max(3, min(6, area - 2)))
        top = _connected_shape(rng, half_h, width, base_n)
        bot = {_refl_row(half_h, r, c) for r, c in top}

        out = np.full((2 * half_h + 1, width), int(bg), dtype=int)
        out[half_h, :] = int(MARK)
        for r, c in top | bot:
            out[int(r), int(c)] = int(color)

        cand: list[tuple[int, int]] = []
        for r in range(half_h):
            for c in range(width):
                if (r, c) in top:
                    continue
                rr, cc = _refl_row(half_h, r, c)
                if int(out[rr, cc]) != int(bg):
                    continue
                if any((int(r + dr), int(c + dc)) in top for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    cand.append((int(r), int(c)))
        if not cand:
            continue

        inp = out.copy()
        extra_n = rng.randint(1, min(2, len(cand)))
        extras = rng.sample(cand, extra_n)
        for r, c in extras:
            inp[int(r), int(c)] = int(color)
        return inp, out, {"type": "row", "extras": int(extra_n)}


def _build_col_patch(rng: random.Random, bg: int, color: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    inp_r, out_r, meta = _build_row_patch(rng, bg, color)
    return inp_r.T.copy(), out_r.T.copy(), {"type": "col", "extras": int(meta["extras"])}


def _transform_offsets(
    offsets: set[tuple[int, int]], rot: int, flip: bool
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in offsets:
        rr, cc = int(r), int(c)
        if flip:
            cc = -cc
        for _ in range(int(rot) % 4):
            rr, cc = -cc, rr
        out.add((int(rr), int(cc)))
    return out


def _build_isolated_patch(rng: random.Random, bg: int, color: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    templates = [
        {(-1, -1), (-2, 0), (-1, 1), (1, 1), (2, 0), (1, -1)},
        {(-1, -1), (-2, 0), (-1, 1), (0, 2), (1, 1), (2, 0), (1, -1), (0, -2)},
        {(-1, -1), (-2, -1), (-2, 0), (-2, 1), (-1, 1), (1, 1), (2, 1), (2, 0), (2, -1), (1, -1)},
    ]
    while True:
        base = set(rng.choice(templates))
        base = _transform_offsets(base, rng.randrange(4), rng.choice([False, True]))

        valid = True
        for r, c in base:
            if (r, c) in {(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)}:
                valid = False
                break
        if not valid:
            continue

        cand: list[tuple[int, int]] = []
        for r in range(-3, 4):
            for c in range(-3, 4):
                if (r, c) in base or (r, c) == (0, 0):
                    continue
                if (r, c) in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
                    continue
                if not (r < 0 or (r == 0 and c < 0)):
                    continue
                if (-r, -c) in base:
                    continue
                if any(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in base):
                    cand.append((int(r), int(c)))
        if not cand:
            continue

        inp = np.full((9, 9), int(bg), dtype=int)
        out = inp.copy()
        center = 4
        inp[center, center] = int(MARK)
        out[center, center] = int(MARK)
        for r, c in base:
            inp[center + int(r), center + int(c)] = int(color)
            out[center + int(r), center + int(c)] = int(color)

        extra_n = rng.randint(1, min(2, len(cand)))
        chosen = rng.sample(cand, extra_n)
        current = set(base)
        for r, c in chosen:
            if not any(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in current):
                continue
            current.add((int(r), int(c)))
            inp[center + int(r), center + int(c)] = int(color)
        if np.count_nonzero(inp != out) == 0:
            continue
        return inp, out, {"type": "isolated", "extras": int(np.count_nonzero(inp != out))}


def _place_patch(
    rng: random.Random,
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    reserve: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    bg: int,
) -> bool:
    ph, pw = map(int, patch_in.shape)
    h, w = map(int, grid_in.shape)
    for _ in range(200):
        r0 = rng.randint(0, h - ph)
        c0 = rng.randint(0, w - pw)
        rr0 = max(0, int(r0 - 1))
        cc0 = max(0, int(c0 - 1))
        rr1 = min(h, int(r0 + ph + 1))
        cc1 = min(w, int(c0 + pw + 1))
        if np.any(reserve[rr0:rr1, cc0:cc1]):
            continue
        sub_in = grid_in[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch_in != int(bg)) & (sub_in != int(bg))):
            continue
        mask = patch_in != int(bg)
        sub_in[mask] = patch_in[mask]
        sub_out = grid_out[r0 : r0 + ph, c0 : c0 + pw]
        sub_out[mask] = patch_out[mask]
        reserve[rr0:rr1, cc0:cc1] = True
        return True
    return False


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()
    palette = [c for c in range(10) if c != int(MARK)]

    for _attempt in range(200):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        bg = rng.choice(palette)
        colors = [c for c in palette if c != int(bg)]
        rng.shuffle(colors)

        grid_in = np.full((h, w), int(bg), dtype=int)
        grid_out = grid_in.copy()
        reserve = np.zeros((h, w), dtype=bool)

        group_types = ["row", "col", "isolated"]
        extra_groups = rng.randint(0, 2)
        for _ in range(extra_groups):
            group_types.append(rng.choice(["row", "col", "isolated"]))

        metas: list[dict[str, int]] = []
        ok = True
        for idx, gtype in enumerate(group_types):
            color = int(colors[idx])
            if gtype == "row":
                patch_in, patch_out, meta = _build_row_patch(rng, bg, color)
            elif gtype == "col":
                patch_in, patch_out, meta = _build_col_patch(rng, bg, color)
            else:
                patch_in, patch_out, meta = _build_isolated_patch(rng, bg, color)
            if not _place_patch(rng, grid_in, grid_out, reserve, patch_in, patch_out, bg):
                ok = False
                break
            metas.append(meta)
        if not ok:
            continue

        solved = main(grid_in)
        if not np.array_equal(solved, grid_out):
            continue

        global LAST_META
        LAST_META = {
            "shape": (int(h), int(w)),
            "bg": int(bg),
            "groups": metas,
            "deleted_pixels": int(np.count_nonzero(grid_in != grid_out)),
        }
        return grid_in

    raise RuntimeError("failed to generate a valid D13 instance")
