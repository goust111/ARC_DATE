from __future__ import annotations
import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4×4 squares into two columns by type: frames left, solids right.

    Input structure:
    - bg = most common color. Multiple 4×4 objects on bg.
    - Frames: colored 1px border, bg-filled 2×2 interior.
    - Solids: entirely filled with one non-bg color.

    Rule:
    - Classify each 4×4 object as frame or solid by its interior.
    - Sort each group top-to-bottom by input row.
    - Stack frames in left column (cols 0–3), solids in right (cols 4–7).
    - Output is 8 wide; height = 4 × max(#frames, #solids), bg-padded.

    Key insight: Frame vs solid is determined solely by whether the 2×2 interior matches bg.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        bcols = set(map(int, border.flatten()))

        if len(bcols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

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

def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(18, 30))
    W = int(np.random.randint(18, 30))
    g = np.full((H, W), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    try:
        used_cols: set[int] = set()
        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            r0, c0 = random_free_location(g, _make_frame(col, bg), transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, _make_frame(col, bg), r0, c0, transparent=bg)
        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            r0, c0 = random_free_location(g, _make_solid(col), transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, _make_solid(col), r0, c0, transparent=bg)
        return g
    except Exception:
        return None

def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")


