from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import STRUCT8, components_mask, colors_present, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network by activating blue blobs via directional green ports.

    Input structure:
    - bg = most common color. Large blue blobs with adjacent green pixels marking port directions.
    - Start marker: yellow pixel atop a short red segment at grid bottom.

    Rule:
    - Extend a ray upward from start; on hitting a blue blob, recolor it red.
    - Activated blobs emit thick rays through green ports until boundary or another blob.
    - Cascade until no new blobs are reached.

    Key insight: Green pixels encode ray direction and width; the network activates as a chain reaction from one start point.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect the start "port" (red vertical segment with a non-red cap above) ---
    yellow_col = None
    red_col = None
    start_col = None
    start_seg: list[tuple[int, int]] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

            # thin vertical segment, height>=2, touches bottom
            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != int(H - 1):
                continue
            # solid vertical segment
            if any(int(g[rr, c0]) != int(col) for rr in range(int(r0), int(r1) + 1)):
                continue
            if int(r0) - 1 < 0:
                continue
            cap = int(g[int(r0) - 1, int(c0)])
            if cap == int(bg) or cap == int(col):
                continue

            red_col = int(col)
            yellow_col = int(cap)
            start_col = int(c0)
            start_seg = [(int(rr), int(c0)) for rr in range(int(r0), int(r1) + 1)]
            break
        if red_col is not None:
            break

    assert not (red_col is None or yellow_col is None or start_col is None)

    # --- STEP 2) Identify blob colors ---
    # Blue = dominant remaining color by pixel count (excluding bg/start colors).
    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    if not other_cols:
        return np.full_like(g, int(bg))
    blue_col = int(max(other_cols)[1])

    # Green = any other color that is 4-adjacent to blue.
    green_col = None
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(blue_col):
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_col = int(col)
            break

    out = np.full_like(g, int(bg))

    # --- STEP 3) Extract blue blobs as connected components, and build a pixel->blob map ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []

    for coords in components_mask(g == int(blue_col), connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = int(len(blobs))
        for r, c in pts:
            blob_id[int(r), int(c)] = bi
        blobs.append({"pts": pts, "ports": []})

    # --- STEP 4) For each blob, find its ports from adjacent green markers ---
    if green_col is not None:
        for blob in blobs:
            pts = blob["pts"]
            seen_dirs: set[tuple[int, int]] = set()
            for r, c in pts:
                for ddr, ddc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = int(r + ddr)
                    cc = int(c + ddc)
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(green_col):
                        seen_dirs.add((int(ddr), int(ddc)))
            for ddr, ddc in seen_dirs:
                port_pixels = [
                    (int(r), int(c))
                    for (r, c) in pts
                    if 0 <= int(r + ddr) < H and 0 <= int(c + ddc) < W and int(g[int(r + ddr), int(c + ddc)]) == int(green_col)
                ]
                if port_pixels:
                    blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        """Recolor a blob to red when first hit by a ray; enqueue it to emit its ports."""
        bi = int(blob_id[int(r), int(c)])
        if bi < 0 or bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    # --- STEP 5) Draw the start ray upward (thin) and stop at the first blob hit ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    if int(top_r) - 1 >= 0 and int(g[int(top_r) - 1, int(start_col)]) == int(yellow_col):
        out[int(top_r) - 1, int(start_col)] = int(red_col)
    for r in range(int(top_r) - 1, -1, -1):
        out[int(r), int(start_col)] = int(red_col)
        if int(g[int(r), int(start_col)]) == int(blue_col):
            activate_blob_at(int(r), int(start_col))
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Extend a ray from each port pixel until boundary or blob hit."""
        ddr, ddc = map(int, d)
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += int(ddr)
                c += int(ddc)
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) == int(blue_col):
                    activate_blob_at(int(r), int(c))
                    break

    # --- STEP 6) Propagate (BFS): activated blob -> emit rays -> activate more blobs ---
    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    return out


