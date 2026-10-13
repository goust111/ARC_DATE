import numpy as np

from arc_utils import DIR4
from arc_utils import bg_boundary as _bg_color, cc4 as _cc4


def _grey_endpoints(g: np.ndarray, grey: int) -> set[tuple[int, int]]:
    h, w = g.shape
    eps: set[tuple[int, int]] = set()
    for r, c in zip(*np.where(g == grey), strict=False):
        r = int(r)
        c = int(c)
        deg = 0
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == grey:
                deg += 1
        if deg <= 1:
            eps.add((r, c))
    return eps


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders (e.g., near the grid edge).
    """
    mask = (g != bg) & (g != grey)
    comps = _cc4(mask)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color == bg or border_color == grey or center_color == grey:
            continue

        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint (er,ec) is a port for this sprite if:
      - it's exactly 2 steps away from the sprite center (Manhattan distance 2),
      - the intermediate cell (1 step away from center toward endpoint) is the sprite border color.
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        ir, ic = cr + dr, cc + dc
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between bordered sprites matched by color, keeping only path-connected ones.

    Input structure:
    - bg, grey (5) path network, and several small "bordered sprites" each having a border color and a single center pixel of a different color.
    - Grey path endpoints touch exactly two destination sprites.

    Rule:
    - Destination sprites are those connected to grey path endpoints.
    - For each destination, find the source sprite whose border color equals the destination's center color.
    - Replace the destination's center pixel with the source's center color.
    - Delete all non-destination sprites; grey paths and destinations remain.

    Key insight: Center color of each destination acts as a lookup key into the border colors of other sprites, enabling indirect color transfer.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    if not sprites:
        return g.copy()

    eps = _grey_endpoints(g, grey)

    # Candidate destinations are sprites that have at least one endpoint-port connection.
    cand = []
    for sp in sprites:
        ports = _ports_for_sprite(sp, g, eps)
        if ports:
            cand.append((sp, ports))

    if not cand:
        return g.copy()

    # There are exactly two destinations; if more candidates exist, prefer smaller center colors.
    cand.sort(key=lambda x: (int(x[0]["center"]), x[0]["center_pos"][0], x[0]["center_pos"][1]))
    dest = [sp for sp, _ in cand[:2]]
    dest_ids = {id(sp) for sp in dest}

    # Build border->sprites lookup (using original input attributes).
    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    # Delete all non-destination sprites completely.
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    # Also delete any remaining non-grey, non-bg pixels not connected to a destination sprite.
    # (Some instances contain extra non-sprite components that are removed in the output.)
    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                if keep[rr, cc]:
                    continue
                if out[rr, cc] == bg or out[rr, cc] == grey:
                    continue
                keep[rr, cc] = True
                q.append((rr, cc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    # Update destination centers by matching their original center color to a source sprite border.
    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        # Deterministic choice if multiple: smallest center color then position.
        srcs_sorted = sorted(srcs, key=lambda s: (int(s["center"]), s["center_pos"][0], s["center_pos"][1]))
        src = srcs_sorted[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out
