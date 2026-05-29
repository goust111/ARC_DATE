# Variant of 2ba387bc
# Dimension: D11 — 干扰项——输入中的额外对象 (src)
# Change: 输入中除4x4方块和背景外无其他对象... → 增加单像素散点噪声

from __future__ import annotations

import numpy as np
from arc_utils import get_bg, paste, random_free_location

# Design analysis for the D11 variant
# ===================================
# 1) Concrete behavior:
#    The core objects stay unchanged: 4x4 hollow frames and 4x4 solid blocks on a
#    uniform background. The new input also contains a few single-pixel colored
#    noise points placed on background cells. At least one noise point is forced
#    to be orthogonally adjacent to a true 4x4 object, so a naive multicolor
#    connected-component pass will merge that pixel into the object's component.
#    Example: a 4x4 solid with one extra side pixel no longer has bbox 4x4 as a
#    component, but the true 4x4 window is still present inside the grid. The new
#    solver therefore searches for valid 4x4 windows directly instead of trusting
#    component bboxes.
#
# 2) Complexity calibration:
#    The variant adds only a small number of pixels (typically 2-5), so the main
#    visual story remains "sort the 4x4 blocks". The attached-noise requirement
#    ensures the change is not degenerate, while isolated noise keeps the input
#    visibly different from the original task. Noise count is kept low so the grid
#    does not become cluttered or introduce many accidental patterns.
#
# 3) Implicit-assumption audit on the original main():
#    - `find_components(... monochromatic=False)`: assumes each object is an entire
#      foreground component. D11 breaks this when a noise pixel touches an object.
#    - `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`: assumes the true object's
#      component bbox remains exactly 4x4. D11 breaks this for attached noise.
#    - `sub = crop_bbox(g, b)`: assumes the crop equals the object. With attached
#      noise the crop includes the spur, so frame/solid checks become invalid.
#    The fix is to detect canonical 4x4 object windows by local shape tests:
#    frame = uniform non-bg border + bg 2x2 interior; solid = all 16 cells same
#    non-bg color. This keeps the task logic unchanged while filtering noise.
#
# 4) Placement legality:
#    Main objects are still placed without overlap. The generator increases object
#    padding to 2 so there is room to add adjacent noise without touching another
#    object. Attached noise is placed in an orthogonally adjacent background cell.
#    Isolated noise is placed only on background cells whose four neighbors are
#    also background, so it remains a single-pixel component.
#
# 5) Visual observability:
#    Across a few examples, a human will see the same sorting rule plus a handful
#    of stray pixels that do not appear in the output. Because at least one stray
#    pixel touches a block in most samples, the variant's behavioral change is
#    visible in the input and matters for solver robustness.
#
# 6) Visual cleanliness:
#    Noise is sparse, objects are well separated, and no object interiors are
#    modified. The 4x4 squares remain easy to perceive as the primary entities.
#
# 7) Color inference:
#    No new color mapping rule is introduced. Noise colors are arbitrary non-bg
#    colors and are ignored. Since output still preserves each object's own color,
#    there is no hidden dependence on numeric color ids.
#
# 8) New assumptions introduced by the revised main():
#    - Every true object still contains an intact 4x4 window matching either the
#      frame template or the solid template.
#    - Two objects do not overlap, so each true object contributes exactly one
#      valid top-left 4x4 window.
#    - Noise never enters the 4x4 footprint of an object; it only occupies former
#      background cells. The generator guarantees all three conditions.


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.tolist()))
    return len(bcols) == 1 and bg not in bcols and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    cols = set(map(int, sub.flatten().tolist()))
    return len(cols) == 1 and bg not in cols


def _classify_window(sub: np.ndarray, bg: int) -> str | None:
    if _is_frame(sub, bg):
        return "frame"
    if _is_solid(sub, bg):
        return "solid"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type, ignoring single-pixel noise.

    D11 change:
    The input may contain extra colored singleton pixels, including pixels attached
    to a square's outer side. Detection therefore scans all 4x4 windows for the
    canonical frame/solid templates instead of depending on component bboxes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for r0 in range(H - 3):
        for c0 in range(W - 3):
            sub = g[r0 : r0 + 4, c0 : c0 + 4]
            kind = _classify_window(sub, bg)
            if kind == "frame":
                frames.append((int(r0), int(c0), sub.copy()))
            elif kind == "solid":
                solids.append((int(r0), int(c0), sub.copy()))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))

    outH = 4 * max(int(len(frames)), int(len(solids)))
    outW = 8
    out = np.full((outH, outW), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _build_expected(bg: int, objects: list[dict]) -> np.ndarray:
    frames = sorted((obj for obj in objects if obj["kind"] == "frame"), key=lambda obj: (obj["r0"], obj["c0"]))
    solids = sorted((obj for obj in objects if obj["kind"] == "solid"), key=lambda obj: (obj["r0"], obj["c0"]))
    out = np.full((4 * max(len(frames), len(solids)), 8), int(bg), dtype=int)
    for i, obj in enumerate(frames):
        paste(out, obj["patch"], i * 4, 0)
    for i, obj in enumerate(solids):
        paste(out, obj["patch"], i * 4, 4)
    return out


def _adjacent_positions(obj: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    cells: list[tuple[int, int]] = []
    for c in range(c0, c0 + 4):
        cells.append((r0 - 1, c))
        cells.append((r0 + 4, c))
    for r in range(r0, r0 + 4):
        cells.append((r, c0 - 1))
        cells.append((r, c0 + 4))
    uniq = []
    seen: set[tuple[int, int]] = set()
    for r, c in cells:
        if 0 <= r < H and 0 <= c < W and (r, c) not in seen:
            seen.add((r, c))
            uniq.append((r, c))
    return uniq


def _isolated_noise_candidates(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    H, W = g.shape
    out: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != int(bg):
                continue
            ok = True
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) != int(bg):
                    ok = False
                    break
            if ok:
                out.append((r, c))
    return out


def _noise_color(used_cols: set[int], avoid: set[int] | None = None) -> int:
    pool = [c for c in range(1, 10) if avoid is None or c not in avoid]
    if not pool:
        pool = list(range(1, 10))
    return int(pool[np.random.randint(0, len(pool))])


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(20, 33))
    W = int(np.random.randint(20, 33))
    g = np.full((H, W), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    objects: list[dict] = []
    try:
        used_cols: set[int] = set()

        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            patch = _make_frame(col, bg)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=2, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            objects.append({"kind": "frame", "r0": int(r0), "c0": int(c0), "patch": patch, "col": int(col)})

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            patch = _make_solid(col)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=2, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            objects.append({"kind": "solid", "r0": int(r0), "c0": int(c0), "patch": patch, "col": int(col)})

        np.random.shuffle(objects)
        attached_target = min(len(objects), int(np.random.randint(1, 4)))
        attached_count = 0
        for obj in objects:
            if attached_count >= attached_target:
                break
            candidates = [(r, c) for r, c in _adjacent_positions(obj, g.shape) if int(g[r, c]) == int(bg)]
            if not candidates:
                continue
            r, c = candidates[int(np.random.randint(0, len(candidates)))]
            noise_col = _noise_color(used_cols, avoid={int(obj["col"])})
            g[r, c] = noise_col
            attached_count += 1

        if attached_count == 0:
            return None

        isolated_candidates = _isolated_noise_candidates(g, bg)
        np.random.shuffle(isolated_candidates)
        isolated_target = int(np.random.randint(1, 4))
        isolated_count = 0
        for r, c in isolated_candidates:
            if isolated_count >= isolated_target:
                break
            g[r, c] = _noise_color(used_cols)
            isolated_count += 1

        expected = _build_expected(bg, objects)
        actual = main(g)
        if actual.shape != expected.shape or not bool((actual == expected).all()):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
